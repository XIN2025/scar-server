import base64
from typing import Dict, Any, Type, List, Union
from cryptography.fernet import Fernet
from pydantic import BaseModel
from pydantic.fields import FieldInfo
import json
from app.config import SECRET_KEY
import hashlib


class EncryptionService:
    def __init__(self):
        if SECRET_KEY:
            self.key = self._generate_fernet_key_from_string(SECRET_KEY)
        else:
            raise ValueError("SECRET_KEY is not set in environment variables.")
        self.cipher_suite = Fernet(self.key)

    def _generate_fernet_key_from_string(self, secret: str) -> bytes:
      digest = hashlib.sha256(secret.encode()).digest()
      return base64.urlsafe_b64encode(digest)

    def _is_encrypted_field(self, field_info: FieldInfo) -> bool:
        return (
            hasattr(field_info, "description")
            and field_info.description == "Encrypted field"
        )

    def _get_encrypted_fields(self, schema_class: Type[BaseModel]) -> set:
        encrypted_fields = set()

        if hasattr(schema_class, "model_fields"):
            for field_name, field_info in schema_class.model_fields.items():
                if self._is_encrypted_field(field_info):
                    encrypted_fields.add(field_name)
        elif hasattr(schema_class, "__fields__"):
            for field_name, field_info in schema_class.__fields__.items():
                if self._is_encrypted_field(field_info):
                    encrypted_fields.add(field_name)

        return encrypted_fields

    def _encrypt_value(self, value: Any) -> str:
        if value is None:
            return None

        json_str = json.dumps(value, default=str)
        encrypted_bytes = self.cipher_suite.encrypt(json_str.encode())
        return base64.b64encode(encrypted_bytes).decode()

    def _decrypt_value(self, encrypted_value: str) -> Any:
        if encrypted_value is None:
            return None

        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
            json_str = decrypted_bytes.decode()
            return json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")

    def _extract_inner_model(self, field_type: Any):
        """
        Return (model_type, is_list) if field_type wraps a BaseModel (directly,
        Optional/Union or List[...] of a BaseModel). Otherwise (None, False).
        """
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return field_type, False

        origin = getattr(field_type, "__origin__", None)
        args = getattr(field_type, "__args__", ())

        # List[Model] or List[Optional[Model]]
        if origin in (list, List) and args:
            inner = args[0]
            # handle Optional inside list
            inner_origin = getattr(inner, "__origin__", None)
            inner_args = getattr(inner, "__args__", ())
            if inner_origin is Union and inner_args:
                for a in inner_args:
                    if isinstance(a, type) and issubclass(a, BaseModel):
                        return a, True
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                return inner, True

        # Optional[Model] (i.e. Union[Model, NoneType]) or other Union
        if origin is Union and args:
            for a in args:
                if isinstance(a, type) and issubclass(a, BaseModel):
                    return a, False

        return None, False

    def encrypt_document(
      self, document: Dict[str, Any], schema_class: Type[BaseModel]
    ) -> Dict[str, Any]:
      if document is None:
        return document
      if isinstance(document, list):
        return [self.encrypt_document(item, schema_class) for item in document]

      input_is_model = isinstance(document, BaseModel)
      doc_dict = document.model_dump() if input_is_model else document.copy()

      encrypted_fields = self._get_encrypted_fields(schema_class)

      for field_name, field_info in getattr(schema_class, "model_fields", {}).items():
        if field_name not in doc_dict:
          continue
        field_type = getattr(field_info, "annotation", None)
        value = doc_dict[field_name]
        if value is None:
          continue

        # handle nested BaseModel, Optional[BaseModel], List[BaseModel], List[Optional[BaseModel]], etc.
        model_inner, is_list = self._extract_inner_model(field_type)
        if model_inner:
          if is_list and isinstance(value, list):
            doc_dict[field_name] = [self.encrypt_document(v, model_inner) for v in value]
          elif isinstance(value, (BaseModel, dict)):
            doc_dict[field_name] = self.encrypt_document(value, model_inner)
          continue

        # fallback: primitive / encrypted field handling
        origin = getattr(field_type, "__origin__", None)
        args = getattr(field_type, "__args__", ())

        if origin in (list, List) and args:
          inner = args[0]
          if isinstance(inner, type) and issubclass(inner, BaseModel) and isinstance(value, list):
            doc_dict[field_name] = [self.encrypt_document(v, inner) for v in value]
          elif field_name in encrypted_fields:
            doc_dict[field_name] = self._encrypt_value(value)
        elif field_name in encrypted_fields:
          doc_dict[field_name] = self._encrypt_value(value)

      return schema_class(**doc_dict) if input_is_model else doc_dict


    def decrypt_document(
      self, encrypted_document: Dict[str, Any], schema_class: Type[BaseModel]
    ) -> Dict[str, Any]:
      if encrypted_document is None:
        return encrypted_document
      if isinstance(encrypted_document, list):
        return [self.decrypt_document(item, schema_class) for item in encrypted_document]

      input_is_model = isinstance(encrypted_document, BaseModel)
      doc_dict = encrypted_document.model_dump() if input_is_model else encrypted_document.copy()

      encrypted_fields = self._get_encrypted_fields(schema_class)

      for field_name, field_info in getattr(schema_class, "model_fields", {}).items():
        if field_name not in doc_dict:
          continue
        field_type = getattr(field_info, "annotation", None)
        value = doc_dict[field_name]
        if value is None:
          continue

        model_inner, is_list = self._extract_inner_model(field_type)
        if model_inner:
          if is_list and isinstance(value, list):
            doc_dict[field_name] = [self.decrypt_document(v, model_inner) for v in value]
          elif isinstance(value, (BaseModel, dict)):
            doc_dict[field_name] = self.decrypt_document(value, model_inner)
          continue

        origin = getattr(field_type, "__origin__", None)
        args = getattr(field_type, "__args__", ())

        if origin in (list, List) and args:
          inner = args[0]
          if isinstance(inner, type) and issubclass(inner, BaseModel) and isinstance(value, list):
            doc_dict[field_name] = [self.decrypt_document(v, inner) for v in value]
          elif field_name in encrypted_fields:
            doc_dict[field_name] = self._decrypt_value(value)
        elif field_name in encrypted_fields:
          doc_dict[field_name] = self._decrypt_value(value)

      return schema_class(**doc_dict) if input_is_model else doc_dict

    def encrypt_documents_bulk(
        self, documents: List[Dict[str, Any]], schema_class: Type[BaseModel]
    ) -> List[Dict[str, Any]]:
        return [self.encrypt_document(doc, schema_class) for doc in documents]

    def decrypt_documents_bulk(
        self, encrypted_documents: List[Dict[str, Any]], schema_class: Type[BaseModel]
    ) -> List[Dict[str, Any]]:
        return [self.decrypt_document(doc, schema_class) for doc in encrypted_documents]

    def get_encryption_info(self, schema_class: Type[BaseModel]) -> Dict[str, Any]:
        encrypted_fields = self._get_encrypted_fields(schema_class)

        return {
            "schema_name": schema_class.__name__,
            "encrypted_fields": list(encrypted_fields),
            "total_encrypted_fields": len(encrypted_fields),
        }


_encryption_service = None

def get_encryption_service() -> EncryptionService:
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service

def set_encryption_service(service: EncryptionService):
    global _encryption_service
    _encryption_service = service
