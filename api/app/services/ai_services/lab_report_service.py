from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import logging
import base64
from datetime import datetime
from bson import ObjectId
import asyncio
import re
import os
import time
from app.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from app.services.backend_services.db import get_db
from app.schemas.ai.lab_report import LabTestProperty, LabReportCreate, LabReportResponse, LabReportSummary, LabReportScore, LabReportScoreGenerate, LabReportExtracted, LabReportPropertyExtracted
from langchain_openai import ChatOpenAI
from app.utils.ai.prompts import get_prompts
from app.schemas.ai.lab_report import LabReport, LabTestPropertyForLLM
from app.services.backend_services.encryption_service import get_encryption_service
from app.schemas.backend.documents import DocumentType
from app.services.backend_services.document_manager import get_document_manager
from app.services.backend_services.progress_tracker import get_progress_tracker
from langchain_core.prompts import ChatPromptTemplate
import fitz 

logger = logging.getLogger(__name__)

class LabReportService:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # TODO: Create a separate instance file for LLM initialization
        # OPTIMIZED: Use fastest model for lab report extraction
        MAX_RETRIES = 2  # Faster failure for retries
        # Use faster model - prioritize speed over quality for extraction
        extraction_model = os.getenv("LAB_REPORT_EXTRACTION_MODEL", None)
        if extraction_model:
            model_to_use = extraction_model
        else:
            # Default to fastest models: gpt-4o-mini (fastest) or gpt-3.5-turbo
            # gpt-4o-mini is 2-3x faster than gpt-4 and much cheaper
            if "gpt-4o-mini" in str(LLM_MODEL).lower():
                model_to_use = LLM_MODEL
            else:
                # Force use of fastest model for extraction
                model_to_use = "gpt-4o-mini"  # Fastest and cheapest option
                logger.info(f"Using gpt-4o-mini for lab report extraction (faster than {LLM_MODEL})")
        
        self.llm = ChatOpenAI(
            model=model_to_use,
            openai_api_key=OPENAI_API_KEY,
            temperature=float(LLM_TEMPERATURE) if LLM_TEMPERATURE else 0.1,  # Lower temp = faster, more deterministic
            max_retries=MAX_RETRIES,
            timeout=60,  # 60 second timeout per request
        )
        self.model = model_to_use  # Store the actual model being used
        self.db = get_db()
        self.collection = self.db["lab_reports"]
        self.prompts = get_prompts()
        self.encryption_service = get_encryption_service()
        self.document_manager = get_document_manager()
        self.progress_tracker = get_progress_tracker()
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, Optional[str]]:
        """Extract metadata from filename using patterns and heuristics"""
        metadata = {
            'test_title': None,
            'test_description': None,
            'lab_name': None,
            'test_date': None
        }
        
        if not filename:
            return metadata
            
        name = filename.replace('.pdf', '').replace('.PDF', '')
        
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  
            r'(\d{2}-\d{2}-\d{4})',  
            r'(\d{2}/\d{2}/\d{4})',  
            r'(\d{4}_\d{2}_\d{2})',   
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, name)
            if match:
                try:
                    date_str = match.group(1).replace('_', '-')
                    # Try to parse and format the date
                    if '-' in date_str:
                        parts = date_str.split('-')
                        if len(parts[0]) == 4:  # YYYY-MM-DD
                            metadata['test_date'] = date_str
                        else:  # MM-DD-YYYY
                            metadata['test_date'] = f"{parts[2]}-{parts[0]}-{parts[1]}"
                    break
                except:
                    continue
        
        # Extract lab names from common patterns
        lab_patterns = [
            r'(LabCorp|Lab Corp)',
            r'(Quest|Quest Diagnostics)',
            r'(ARUP|Arup)',
            r'(Mayo|Mayo Clinic)',
            r'(Cleveland|Cleveland Clinic)',
            r'(Johns Hopkins)',
            r'(Lab|Laboratory)',
        ]
        
        for pattern in lab_patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                metadata['lab_name'] = match.group(1)
                break
        
        # Generate title from filename
        if name:
            clean_name = re.sub(r'[_\-\d]+', ' ', name)
            clean_name = re.sub(r'\s+', ' ', clean_name).strip()
            
            words = clean_name.split()[:4]  
            if words:
                title = ' '.join(words)
                metadata['test_title'] = title
                metadata['test_description'] = f"Lab report from {clean_name}"
        
        return metadata
    
    def _generate_fallback_metadata(self, filename: str, upload_date: datetime) -> Dict[str, str]:
        """Generate comprehensive fallback metadata"""
        fallback = {
            'test_title': 'Lab Report',
            'test_description': 'Medical laboratory test results',
            'test_date': upload_date.strftime('%Y-%m-%d'),
            'lab_name': None,
            'doctor_name': None
        }
        
        # Try to extract from filename first
        filename_metadata = self._extract_metadata_from_filename(filename)
        
        # Use filename metadata if available
        if filename_metadata['test_title']:
            fallback['test_title'] = filename_metadata['test_title']
        if filename_metadata['test_description']:
            fallback['test_description'] = filename_metadata['test_description']
        if filename_metadata['test_date']:
            fallback['test_date'] = filename_metadata['test_date']
        if filename_metadata['lab_name']:
            fallback['lab_name'] = filename_metadata['lab_name']
        
        # Generate unique title if still generic
        if fallback['test_title'] == 'Lab Report':
            timestamp = upload_date.strftime('%Y%m%d_%H%M')
            fallback['test_title'] = f"Lab Report {timestamp}"
        
        return fallback
    
    def _validate_and_clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted metadata"""
        cleaned = {}
        
        title = metadata.get('test_title')
        if title and isinstance(title, str):
            title = title.strip()
            if len(title) > 3 and len(title) < 100:
                cleaned['test_title'] = title
            else:
                cleaned['test_title'] = None
        else:
            cleaned['test_title'] = None
        
        # Clean and validate test_description
        description = metadata.get('test_description')
        if description and isinstance(description, str):
            description = description.strip()
            if len(description) > 5 and len(description) < 500:
                cleaned['test_description'] = description
            else:
                cleaned['test_description'] = None
        else:
            cleaned['test_description'] = None
        
        # Clean and validate test_date
        date_str = metadata.get('test_date')
        if date_str and isinstance(date_str, str):
            date_str = date_str.strip()
            if date_str:
                try:
                    # Try to parse the date
                    parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
                    # Check if date is reasonable (not too far in future/past)
                    current_year = datetime.now().year
                    if 1900 <= parsed_date.year <= current_year + 1:
                        cleaned['test_date'] = date_str
                    else:
                        cleaned['test_date'] = None
                except:
                    cleaned['test_date'] = None
            else:
                cleaned['test_date'] = None
        else:
            cleaned['test_date'] = None
        
        # Clean lab_name
        lab_name = metadata.get('lab_name')
        if lab_name and isinstance(lab_name, str):
            lab_name = lab_name.strip()
            if len(lab_name) > 2 and len(lab_name) < 100:
                cleaned['lab_name'] = lab_name
            else:
                cleaned['lab_name'] = None
        else:
            cleaned['lab_name'] = None
        
        # Clean doctor_name
        doctor_name = metadata.get('doctor_name')
        if doctor_name and isinstance(doctor_name, str):
            doctor_name = doctor_name.strip()
            if len(doctor_name) > 2 and len(doctor_name) < 100:
                cleaned['doctor_name'] = doctor_name
            else:
                cleaned['doctor_name'] = None
        else:
            cleaned['doctor_name'] = None
        
        return cleaned
    
    def _merge_metadata(self, extracted: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        """Merge extracted metadata with fallback, preferring extracted when valid"""
        merged = {}
        
        for key in ['test_title', 'test_description', 'test_date', 'lab_name', 'doctor_name']:
            extracted_val = extracted.get(key)
            fallback_val = fallback.get(key)
            
            # Use extracted value if it's valid, otherwise use fallback
            if extracted_val and isinstance(extracted_val, str) and extracted_val.strip():
                merged[key] = extracted_val
            elif fallback_val and isinstance(fallback_val, str) and fallback_val.strip():
                merged[key] = fallback_val
            else:
                merged[key] = None
        
        return merged
    
    async def _invoke_structured_llm(
        self, schema: dict, system_prompt: str, user_prompt: str, input_vars: dict
    ) -> dict:
        """
        Invoke structured LLM - OPTIMIZED: Use fastest model (gpt-4o-mini) with timeout
        """
        # Use the same optimized model from initialization (defaults to gpt-4o-mini)
        extraction_model = os.getenv("LAB_REPORT_EXTRACTION_MODEL", self.model)
        llm = ChatOpenAI(
            model=extraction_model,
            openai_api_key=OPENAI_API_KEY,
            temperature=0, 
            timeout=60,  
            max_retries=2 
        ).with_structured_output(schema)
        
        try:
            if not input_vars:
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                result = await llm.ainvoke(messages)
            else:
                prompt = ChatPromptTemplate.from_messages(
                    [("system", system_prompt), ("user", user_prompt)]
                )
                chain = prompt | llm
                result = await chain.ainvoke(input_vars)
            
            if not isinstance(result, dict):
                logger.error(f"_invoke_structured_llm returned non-dict type: {type(result).__name__}, value: {result}")
                return {}
            
            return result
        except Exception as e:
            logger.error(f"Error in _invoke_structured_llm: {type(e).__name__}: {str(e)}")
            raise

    def _convert_pdf_bytes_to_markdown(self, pdf_bytes: bytes) -> Optional[str]:
        """Convert PDF bytes to markdown - synchronous method for thread pool execution"""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            md_lines = []
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                md_lines.append(f"\n## Page {page_num}\n\n{text}")
        markdown = '\n'.join(md_lines)
        logger.debug("Converted PDF bytes to markdown string")
        return markdown
        
    async def _rewrite_extracted_markdown(self, markdown: str) -> Optional[str]:
        system_prompt = (
            "You are an expert medical document editor. "
            "Rewrite the provided markdown content to improve clarity, "
            "fix formatting issues, and ensure medical terminology is used correctly. "
            "PRESERVE ALL MEDICAL DATA including symbols, values, units, and special characters. "
            "Do NOT remove or filter any medical information."
        )
        user_prompt = (
            "Here is the extracted markdown content from a lab report:\n\n"
            "{markdown}\n\n"
            "Please provide a cleaned-up version of this markdown."
        )
        schema = {
            "title": "RewrittenMarkdown",
            "description": "A cleaned-up version of the extracted markdown from a medical lab report.",
            "type": "object",
            "properties": {
                "rewritten_markdown": {"type": "string"}
            },
            "required": ["rewritten_markdown"]
        }
        input_vars = {"markdown": markdown}
        
        try:
            result = await self._invoke_structured_llm(
                schema, system_prompt, user_prompt, input_vars
            )
            rewritten_md = result.get("rewritten_markdown", None)
            print(f"Rewrote extracted markdown")
            return rewritten_md
        except Exception as e:
            print(f"Error rewriting markdown: {e}")
            return None

    def _split_text_into_chunks(self, text: str, chunk_size: int = 50000) -> List[str]:
        """
        Split text into chunks - OPTIMIZED: Much larger chunk size to minimize LLM calls
        Larger chunks = fewer API calls = faster processing
        """
        if not text:
            return []

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current = []

        def flush_current():
            if current:
                chunks.append("".join(current).strip())
                current.clear()

        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para if para.endswith("\n\n") else para + "\n\n"
            # If paragraph fits in the current buffer
            if sum(len(p) for p in current) + len(para) <= chunk_size:
                current.append(para)
                continue

            # If paragraph itself is small but current has content, flush current first
            if current:
                flush_current()

            # If paragraph fits into an empty chunk, take it
            if len(para) <= chunk_size:
                current.append(para)
                continue

            # Paragraph is larger than chunk_size: split by words
            words = para.split()
            buf = []
            for w in words:
                # add space before words when needed
                add = ((" " if buf else "") + w)
                if sum(len(x) for x in buf) + len(add) > chunk_size:
                    # flush buf to chunks
                    chunks.append("".join(buf).strip())
                    buf = [w]
                else:
                    buf.append(add)
            if buf:
                chunks.append("".join(buf).strip())

        # flush any remaining content
        flush_current()
        return chunks

    async def _process_chunk(self, chunk: str, schema: dict, prompt: str) -> tuple[dict, str]:
        """
        Process a chunk - OPTIMIZED: Skip markdown rewriting to save 50% LLM calls
        The extracted markdown from PDF is usually clean enough for direct extraction
        """
        try:
            # SKIP markdown rewriting - this was doubling our LLM calls!
            # The PDF extraction already produces clean enough markdown
            extracted_properties = await self._invoke_structured_llm(
                schema=schema,
                system_prompt=prompt,
                user_prompt=chunk,  # Use chunk directly, no rewriting
                input_vars={}
            )
            if not isinstance(extracted_properties, dict):
                logger.error(f"Expected dict from _invoke_structured_llm, got {type(extracted_properties).__name__}: {extracted_properties}")
                return None, chunk
            return extracted_properties, chunk
        except Exception as e:
            chunk_preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            logger.warning(
                f"Failed to process chunk (length: {len(chunk)} chars, preview: {chunk_preview}): "
                f"{type(e).__name__}: {str(e)}"
            )

            if len(chunk) > 8000:
                mid = len(chunk) // 2
                subchunks = [chunk[:mid], chunk[mid:]]
                combined_result: dict[str, Any] = {
                    "properties": [],
                    "lab_name": None,
                    "doctor_name": None,
                    "test_title": None,
                    "test_description": None,
                    "test_date": None,
                }
                recovered_props = 0

                for idx, subchunk in enumerate(subchunks, start=1):
                    sub_result, _ = await self._process_chunk(subchunk, schema, prompt)
                    if not sub_result:
                        logger.warning(
                            f"Fallback chunk {idx}/{len(subchunks)} (length: {len(subchunk)} chars) also failed"
                        )
                        continue
                    
                    if not isinstance(sub_result, dict):
                        logger.error(f"Fallback chunk {idx} returned invalid type: {type(sub_result).__name__}, expected dict")
                        continue

                    recovered_props += len(sub_result.get("properties", []))
                    combined_result["properties"].extend(sub_result.get("properties", []))

                    for key in ["lab_name", "doctor_name", "test_title", "test_description", "test_date"]:
                        if sub_result.get(key) and not combined_result.get(key):
                            combined_result[key] = sub_result.get(key)

                if recovered_props > 0:
                    logger.info(
                        f"Recovered {recovered_props} properties by splitting chunk of length {len(chunk)} "
                        f"into {len(subchunks)} sub-chunks"
                    )
                    return combined_result, chunk

            logger.error(
                f"Exhausted retries for chunk (length: {len(chunk)}). No properties recovered. "
                f"Last error: {type(e).__name__}: {str(e)}"
            )
            return None, chunk

    async def _extract_lab_data_from_markdown(self, markdown: str, filename: str = None) -> LabReportExtracted:
        """
        Extract lab data - OPTIMIZED: More concise prompt for faster processing
        """
        prompt = """Extract lab test data. Return JSON with:
- test_title (3-5 words), test_description, test_date (YYYY-MM-DD)
- properties: Array of {{property_name, value, unit, reference_range, status}}
- lab_name, doctor_name if available

CRITICAL RULES - READ CAREFULLY:
1. Extract EVERY SINGLE test result from the document - do not skip any
2. If you see a test name with a value, it MUST be included in the properties array
3. property_name and value are REQUIRED - never null/empty
4. Preserve ALL medical symbols (%, &, <, >, etc.) exactly as shown
5. Use exact test names as they appear in the document
6. Include reference ranges, units, status exactly as shown
7. Count all test results - aim for 20-50+ properties per chunk for comprehensive reports
8. If the document has many tests, extract ALL of them - completeness is more important than speed
9. Do not summarize or combine similar tests - each test result should be a separate property

IMPORTANT: This is a medical document. Missing test results could be critical. Extract EVERYTHING."""

        schema = {
            "title": "LabReport",
            "description": "Extracted structured lab test data from a medical lab report PDF.",
            "type": "object",
            "properties": {
                "properties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "property_name": {"type": "string"},
                            "value": {"type": "string"},
                            "unit": {"type": "string"},
                            "reference_range": {"type": "string"},
                            "status": {"type": "string"}
                        },
                        "required": ["property_name", "value"]
                    }
                },
                "test_title": {"type": "string"},
                "test_description": {"type": "string"},
                "test_date": {"type": "string"},
                "lab_name": {"type": "string"},
                "doctor_name": {"type": "string"}
            },
            "required": ["properties"]
        }


        # Initialize metadata tracking
        extracted_metadata = {
            'lab_name': None,
            'doctor_name': None,
            'test_title': None,
            'test_description': None,
            'test_date': None
        }
        
        chunk_size = int(os.getenv("LAB_REPORT_CHUNK_SIZE", "15000"))
        chunks = self._split_text_into_chunks(markdown, chunk_size)
        logger.info(f"⏱️ Split markdown into {len(chunks)} chunks (chunk_size: {chunk_size}, total length: {len(markdown)})")
        
        chunks_to_process = chunks
        logger.info(f"⏱️ Processing all {len(chunks)} chunks to extract all properties")
        
        properties = []
        
        # Time the chunk processing
        chunk_start = time.time()
        chunk_tasks = [
            self._process_chunk(chunk, schema, prompt)
            for chunk in chunks_to_process
        ]
        
        chunk_results = await asyncio.gather(*chunk_tasks)
        chunk_time = time.time() - chunk_start
        logger.info(f"⏱️ LLM chunk processing took: {chunk_time:.2f}s ({len(chunks_to_process)} chunks)")
        
        failed_chunks = 0
        successful_chunks = 0
        seen_properties = set()  
        
        for idx, (extracted_properties, _) in enumerate(chunk_results):
            if not extracted_properties:
                failed_chunks += 1
                logger.warning(f"Chunk {idx + 1}/{len(chunks_to_process)} failed to extract properties")
                continue
            
            if not isinstance(extracted_properties, dict):
                failed_chunks += 1
                logger.error(f"Chunk {idx + 1}/{len(chunks_to_process)} returned invalid type: {type(extracted_properties).__name__}, expected dict")
                continue
            
            successful_chunks += 1
            chunk_props_count = 0
            properties_list = extracted_properties.get("properties", [])
            raw_props_count = len(properties_list) if isinstance(properties_list, (list, dict)) else 0
            
            if isinstance(properties_list, dict):
                logger.warning(f"Chunk {idx + 1} returned properties as dict (keys: {list(properties_list.keys())[:5]}), attempting to convert")
                if "property_name" in properties_list or "value" in properties_list:
                    properties_list = [properties_list]
                    logger.info(f"Chunk {idx + 1}: Converted single property dict to list")
                elif all(isinstance(v, dict) for v in properties_list.values() if v):
                    properties_list = [v for v in properties_list.values() if isinstance(v, dict)]
                    logger.info(f"Chunk {idx + 1}: Converted nested dict with {len(properties_list)} properties to list")
                else:
                    logger.error(f"Chunk {idx + 1}: Cannot convert dict structure to list, skipping chunk")
                    continue
            elif not isinstance(properties_list, list):
                logger.error(f"Chunk {idx + 1} properties is not a list or dict: {type(properties_list).__name__}, skipping chunk")
                continue
            
            skipped_missing = 0
            skipped_duplicates = 0
            skipped_invalid = 0
                
            for prop in properties_list:
                if not isinstance(prop, dict):
                    skipped_invalid += 1
                    continue
                    
                property_name = prop.get("property_name")
                value = prop.get("value")
                unit = prop.get("unit")
                reference_range = prop.get("reference_range")
                status = prop.get("status")
                
                if not property_name or not value:
                    skipped_missing += 1
                    continue
                
                prop_key = f"{property_name}|{value}".lower().strip()
                if prop_key in seen_properties:
                    skipped_duplicates += 1
                    continue
                
                seen_properties.add(prop_key)
                
                try:
                    properties.append(LabReportPropertyExtracted(
                        property_name=property_name,
                        value=value,
                        unit=unit,
                        reference_range=reference_range,
                        status=status
                    ))
                    chunk_props_count += 1
                except Exception as e:
                    skipped_invalid += 1
                    continue
            
            if skipped_missing > 0 or skipped_duplicates > 0 or skipped_invalid > 0:
                logger.debug(f"Chunk {idx + 1}: raw={raw_props_count}, extracted={chunk_props_count}, skipped_missing={skipped_missing}, skipped_duplicates={skipped_duplicates}, skipped_invalid={skipped_invalid}")
            logger.info(f"Chunk {idx + 1}/{len(chunks_to_process)} extracted {chunk_props_count} properties (total so far: {len(properties)})")
            
            # Aggregate metadata from this chunk (only if not already set)
            chunk_metadata = {
                'lab_name': extracted_properties.get("lab_name"),
                'doctor_name': extracted_properties.get("doctor_name"),
                'test_title': extracted_properties.get("test_title"),
                'test_description': extracted_properties.get("test_description"),
                'test_date': extracted_properties.get("test_date")
            }
            for key, value in chunk_metadata.items():
                # normalize and avoid KeyError if key missing
                value_stripped = value.strip() if isinstance(value, str) else value
                if value_stripped and not extracted_metadata.get(key):
                    extracted_metadata[key] = value_stripped    

        # Generate fallback metadata
        upload_date = datetime.now()
        fallback_metadata = self._generate_fallback_metadata(filename or "lab-report.pdf", upload_date)
        
        # Validate and clean extracted metadata
        cleaned_extracted = self._validate_and_clean_metadata(extracted_metadata)
        
        # Merge extracted with fallback
        final_metadata = self._merge_metadata(cleaned_extracted, fallback_metadata)
        
        logger.info(f"📊 Extraction Summary: {len(properties)} total properties from {successful_chunks}/{len(chunks_to_process)} chunks")
        if failed_chunks > 0:
            logger.warning(f"⚠️ {failed_chunks} chunks failed - some properties may be missing")
        
        # Handle case where no properties were extracted
        if not properties:
            logger.warning("No valid properties extracted from lab report")
            properties = [
                LabReportPropertyExtracted(
                    property_name="No Data Extracted",
                    value="Unable to extract lab data from this document",
                    test_title=final_metadata.get('test_title', 'Lab Report'),
                    test_description=final_metadata.get('test_description', 'Lab Report')
                )
            ]

        result = LabReportExtracted(
            properties=properties,
            lab_name=final_metadata.get('lab_name'),
            doctor_name=final_metadata.get('doctor_name'),
            test_title=final_metadata.get('test_title'),
            test_description=final_metadata.get('test_description'),
            test_date=final_metadata.get('test_date')
        )

        logger.info(f"Successfully extracted {len(properties)} properties from lab report")
        return result

    async def process_lab_report_pdf(self, file_content: bytes, filename: str, user_email: str, document_type: DocumentType, upload_id: Optional[str] = None) -> LabReportResponse:
        """Process a lab report PDF and extract structured data - optimized with parallel operations"""
        total_start = time.time()
        try:
            logger.info(f"Starting lab report processing for file: {filename}")
            
            # Create upload session if not provided
            if not upload_id:
                upload_id = self.progress_tracker.create_upload_session(filename)
            
            # Update progress
            self.progress_tracker.update_progress(upload_id, 10, "Converting PDF to text...")
            
            # Convert PDF to markdown in thread pool (non-blocking)
            pdf_start = time.time()
            markdown = await asyncio.to_thread(
                self._convert_pdf_bytes_to_markdown,
                file_content
            )
            pdf_time = time.time() - pdf_start
            logger.info(f"⏱️ PDF conversion took: {pdf_time:.2f}s")
            
            if not markdown:
                raise ValueError("Unable to extract text from PDF. The document may be corrupted or unsupported.")

            logger.info(f"Successfully extracted markdown from PDF, length: {len(markdown)}")
            self.progress_tracker.update_progress(upload_id, 30, "Extracting lab data with AI...")
            
            # Extract lab data from markdown
            llm_start = time.time()
            lab_report = await self._extract_lab_data_from_markdown(markdown, filename)
            llm_time = time.time() - llm_start
            logger.info(f"⏱️ LLM extraction took: {llm_time:.2f}s")
            logger.info(f"Successfully extracted lab data, properties count: {len(lab_report.properties)}")
            
            self.progress_tracker.update_progress(upload_id, 60, "Saving lab report and creating embeddings...")

            # Parse test_date if it's a string
            parsed_test_date = None
            if lab_report.test_date and isinstance(lab_report.test_date, str):
                try:
                    parsed_test_date = datetime.strptime(lab_report.test_date, '%Y-%m-%d')
                except Exception as e:
                    logger.warning(f"Could not parse test_date: {lab_report.test_date}, error: {e}")
                    parsed_test_date = None

            # Ensure all string fields are safe
            safe_test_title = None
            if lab_report.test_title and isinstance(lab_report.test_title, str):
                safe_test_title = lab_report.test_title.strip()
                if not safe_test_title:
                    safe_test_title = None
            
            safe_test_description = None
            if lab_report.test_description and isinstance(lab_report.test_description, str):
                safe_test_description = lab_report.test_description.strip()
                if not safe_test_description:
                    safe_test_description = None
            
            safe_lab_name = None
            if lab_report.lab_name and isinstance(lab_report.lab_name, str):
                safe_lab_name = lab_report.lab_name.strip()
                if not safe_lab_name:
                    safe_lab_name = None
            
            safe_doctor_name = None
            if lab_report.doctor_name and isinstance(lab_report.doctor_name, str):
                safe_doctor_name = lab_report.doctor_name.strip()
                if not safe_doctor_name:
                    safe_doctor_name = None

            # Create lab report object with comprehensive fallbacks
            lab_report_data = LabReportCreate(
                user_email=user_email,
                test_title=safe_test_title or f"Lab Report {datetime.now().strftime('%Y%m%d_%H%M')}",
                test_description=safe_test_description or f"Medical laboratory test results from {filename}",
                properties=lab_report.properties,
                test_date=parsed_test_date,
                lab_name=safe_lab_name,
                doctor_name=safe_doctor_name,
                upload_id=upload_id
            )

            logger.info(f"Document ID: {lab_report_data.upload_id}")

            # OPTIMIZED: Save to database first (required), vector store in background (non-blocking)
            save_start = time.time()
            saved_report = await self.save_lab_report(lab_report_data, filename)
            save_time = time.time() - save_start
            logger.info(f"⏱️ Database save took: {save_time:.2f}s")
            
            # Start vector store processing in background (don't wait - non-blocking)
            vector_task = asyncio.create_task(
                asyncio.to_thread(
                    self.document_manager.add_document,
                    content=markdown,  # Reuse markdown instead of re-processing PDF
                    filename=filename,
                    user_email=user_email,
                    upload_id=upload_id,
                    type=document_type,
                )
            )
            # Don't await - let it run in background
            logger.info("⏱️ Vector store processing started in background (non-blocking)")
            
            self.progress_tracker.update_progress(upload_id, 100, "Lab report processed successfully", "completed")
            
            total_time = time.time() - total_start
            logger.info(f"⏱️ TOTAL processing time: {total_time:.2f}s (PDF: {pdf_time:.2f}s, LLM: {llm_time:.2f}s, DB: {save_time:.2f}s)")
            logger.info(f"Successfully saved lab report with ID: {saved_report.id}")
            return saved_report
            
        except Exception as e:
            total_time = time.time() - total_start if 'total_start' in locals() else 0
            logger.error(f"Error processing lab report PDF: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"⏱️ Failed after: {total_time:.2f}s")
            
            # Update progress tracker on error
            if upload_id:
                self.progress_tracker.update_progress(
                    upload_id, 
                    100, 
                    f"Processing failed: {str(e)}", 
                    "failed",
                    error_message=str(e)
                )
            
            try:
                logger.info("Creating fallback lab report due to processing error")
                fallback_report = LabReportCreate(
                    user_email=user_email,
                    test_title=f"Lab Report {datetime.now().strftime('%Y%m%d_%H%M')}",
                    test_description=f"Lab report from {filename} (processing had issues)",
                    properties=[
                        LabReportPropertyExtracted(
                            property_name="Processing Error",
                            value="Unable to fully process this lab report",
                            test_title=f"Lab Report {datetime.now().strftime('%Y%m%d_%H%M')}",
                            test_description=f"Lab report from {filename}"
                        )
                    ],
                    test_date=datetime.now(),
                    lab_name=None,
                    doctor_name=None,
                    upload_id=upload_id or ("fallback-" + datetime.now().strftime('%Y%m%d%H%M%S'))
                )
                
                saved_report = await self.save_lab_report(fallback_report, filename)
                logger.info(f"Successfully created fallback report with ID: {saved_report.id}")
                return saved_report
                
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback report: {str(fallback_error)}")
                raise Exception(f"Lab report processing failed: {str(e)}. Fallback also failed: {str(fallback_error)}")

    async def delete_lab_report_by_upload_id(self, upload_id: str, user_email: str) -> bool:
        try:
            # Delete from vector store first to avoid orphaned vectors
            self.document_manager.delete_document_by_upload_id(upload_id)
            # Then delete from MongoDB
            result = await self.collection.delete_one({
                "upload_id": upload_id,
                "user_email": user_email
            })
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting lab report: {str(e)}")
            return False

    async def save_lab_report(self, lab_report: LabReportCreate, filename: str) -> LabReportResponse:
        """Save lab report to database"""
        try:
            lab_report = self.encryption_service.encrypt_document(lab_report, LabReportCreate)
            # Prepare document for MongoDB
            doc = {
                "user_email": lab_report.user_email,
                "test_title": lab_report.test_title,
                "test_description": lab_report.test_description,
                "properties": [prop.model_dump() for prop in lab_report.properties],
                "test_date": lab_report.test_date,
                "lab_name": lab_report.lab_name,
                "doctor_name": lab_report.doctor_name,
                "filename": filename,
                "upload_id": lab_report.upload_id,
                "created_at": datetime.utcnow()
            }
            
            # Insert into database
            result = await self.collection.insert_one(doc)
            
            # Return response with generated ID
            return LabReportResponse(
                id=str(result.inserted_id),
                user_email=lab_report.user_email,
                test_title=lab_report.test_title,
                test_description=lab_report.test_description,
                properties=lab_report.properties,
                test_date=lab_report.test_date,
                lab_name=lab_report.lab_name,
                doctor_name=lab_report.doctor_name,
                filename=filename,
                created_at=doc["created_at"]
            )
            
        except Exception as e:
            logger.error(f"Error saving lab report: {str(e)}")
            raise

    # TODO: Instead of returning a list of LabReportSummary, I think it would be better a list of LabReport 
    async def get_lab_reports_by_user(self, user_email: str) -> List[LabReportSummary]:
        """Get all lab reports for a user (summary view)"""
        try:
            cursor = await self.collection.find(
                {"user_email": user_email},
                {
                    "test_title": 1,
                    "test_description": 1,
                    "test_date": 1,
                    "filename": 1,
                    "created_at": 1,
                    "properties": 1,
                    "upload_id": 1
                }
            ).sort("created_at", -1).to_list(length=None)
            
            reports = []
            for doc in cursor:
                logger.debug(f"Processing lab report: id={doc.get('_id')}, title={doc.get('test_title', 'N/A')[:50]}, properties_count={len(doc.get('properties', []))}")
                reports.append(LabReportSummary(
                    id=str(doc["_id"]),
                    test_title=doc.get("test_title", ""),
                    test_description=doc.get("test_description", ""),
                    test_date=doc.get("test_date"),
                    properties_count=len(doc.get("properties", [])),
                    filename=doc.get("filename", ""),
                    created_at=doc.get("created_at", datetime.utcnow()),
                    upload_id=doc.get("upload_id", "")
                ))
    
            reports = self.encryption_service.decrypt_document(reports, LabReportSummary)

            return reports

        except Exception as e:
            logger.error(f"Error fetching lab reports: {str(e)}")
            raise

    async def get_lab_report_context_by_user(self, user_email: str) -> str:
        lab_reports = await self.get_lab_reports_by_user(user_email)
        context = "User Lab Reports Summary:\n"
        for report in lab_reports:
            context += (
                f"- Title: {report.test_title}\n"
                f"  Date: {report.test_date.strftime('%Y-%m-%d') if report.test_date else 'N/A'}\n"
                f"  Description: {report.test_description}\n"
                f"  Properties Count: {report.properties_count}\n"
                f"  Filename: {report.filename}\n"
                f"  Created At: {report.created_at.strftime('%Y-%m-%d %H:%M:%S') if report.created_at else 'N/A'}\n"
            )
        return context

    # TODO: Instead of returning LabReportResponse, I think it would be better to return LabReport
    async def get_lab_report_by_id(self, report_id: str, user_email: str) -> Optional[LabReportResponse]:
        """Get detailed lab report by ID"""
        try:
            # Validate ObjectId format
            if not ObjectId.is_valid(report_id):
                return None
            
            doc = await self.collection.find_one({
                "_id": ObjectId(report_id),
                "user_email": user_email
            })
            
            if not doc:
                return None
            
            # Convert properties back to LabTestProperty objects
            properties = []
            for prop_data in doc.get("properties", []):
                try:
                    if prop_data.get("property_name") and prop_data.get("value"):
                        properties.append(LabReportPropertyExtracted(**prop_data))
                    else:
                        logger.warning(f"Skipping invalid property data: {prop_data}")
                except Exception as e:
                    logger.warning(f"Failed to create LabReportPropertyExtracted from data: {prop_data}, error: {e}")
                    continue

            properties = [self.encryption_service.decrypt_document(prop, LabReportPropertyExtracted) for prop in properties]

            return LabReportResponse(
                id=str(doc["_id"]),
                user_email=doc.get("user_email", ""),
                test_title=doc.get("test_title", ""),
                test_description=doc.get("test_description", ""),
                properties=properties,
                test_date=doc.get("test_date"),
                lab_name=doc.get("lab_name"),
                doctor_name=doc.get("doctor_name"),
                filename=doc.get("filename", ""),
                created_at=doc.get("created_at", datetime.utcnow())
            )
            
        except Exception as e:
            logger.error(f"Error fetching lab report by ID: {str(e)}")
            raise
    
    async def delete_lab_report_by_filename(self, filename: str, user_email: str) -> bool:
        """Delete a lab report by filename and user email"""
        try:
            result = await self.collection.delete_one({
                "filename": filename,
                "user_email": user_email
            })
            
            if result.deleted_count == 0:
                logger.warning(f"Lab report with filename '{filename}' not found or doesn't belong to user {user_email}")
                return False
            
            logger.info(f"Successfully deleted lab report '{filename}' for user {user_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting lab report by filename '{filename}': {str(e)}")
            raise

    async def score_lab_report(self, lab_report: LabReport) -> LabReportScore:
        prompt = self.prompts.get_lab_report_score_prompt(lab_report)
        structured_llm = self.llm.with_structured_output(LabReportScoreGenerate)
        response = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
        lab_report_score = LabReportScore(
            # TODO: Clearly a schema issue that needs to be fixed in lab reports
            lab_report_id="",  
            score=response.score,
            reasons=response.reasons
        )
        return lab_report_score

# Global instance
_lab_report_service = None

def get_lab_report_service() -> LabReportService:
    """Get or create a LabReportService instance"""
    global _lab_report_service
    if _lab_report_service is None:
        _lab_report_service = LabReportService()
    return _lab_report_service

