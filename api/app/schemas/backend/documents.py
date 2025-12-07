from enum import Enum

class DocumentType(str, Enum):
  DOCUMENT = "document"
  LAB_REPORT = "lab_report"