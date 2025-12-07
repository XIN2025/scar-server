from pydantic import Field

def EncryptedField(*args, **kwargs):
    """Custom field to mark encrypted data"""
    kwargs.setdefault('description', 'Encrypted field')
    return Field(*args, **kwargs)