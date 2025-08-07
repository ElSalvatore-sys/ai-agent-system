from __future__ import annotations

"""Utilities for cryptographically verifying local LLM model files.

Only detached signature verification is supported for now. The signer is
expected to create a detached Ed25519 signature for the *raw* model file
(`model.bin`, `model.safetensors`, etc.) and distribute a corresponding
`model.bin.sig` (or similar) alongside the file.

Public-verification keys are kept in the application *trust store* – a local
read-only directory that should be provisioned at install time or mounted from
secure storage (e.g. TPM-sealed files, HSM, or a secrets manager volume).

Example usage
-------------
>>> from pathlib import Path
>>> from app.security.model_verifier import verify_model_signature
>>> ok = verify_model_signature(
...     model_path=Path("models/llama2.safetensors"),
...     signature_path=Path("models/llama2.safetensors.sig"),
...     public_key_path=Path("trust_store/llama2.pub"),
... )
>>> print("verified" if ok else "INVALID MODEL – quarantine")
"""

import hashlib
import logging
from pathlib import Path
from typing import Final

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.exceptions import InvalidSignature

# ---------------------------------------------------------------------------
# Constants & logger
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE: Final[int] = 4 * 1024 * 1024  # 4 MiB
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def verify_model_signature(
    model_path: Path | str,
    signature_path: Path | str,
    public_key_path: Path | str,
) -> bool:
    """Verify the detached Ed25519 signature of *model_path*.

    Parameters
    ----------
    model_path:
        Path to the model weights file (e.g., .bin, .pt, .safetensors).
    signature_path:
        Path to the detached signature produced with the signer’s **private**
        Ed25519 key. The file must contain the raw 64-byte signature, *not*
        ASCII-armored.
    public_key_path:
        Path to the PEM-encoded **public** Ed25519 key that corresponds to the
        private signing key.

    Returns
    -------
    bool
        ``True`` if signature *and* hash are valid, otherwise ``False``.
    """

    model_path = Path(model_path)
    signature_path = Path(signature_path)
    public_key_path = Path(public_key_path)

    if not model_path.is_file():
        logger.error("Model file missing: %s", model_path)
        return False
    if not signature_path.is_file():
        logger.error("Signature file missing: %s", signature_path)
        return False
    if not public_key_path.is_file():
        logger.error("Public-key file missing: %s", public_key_path)
        return False

    try:
        # 1. Read signature (raw bytes, 64 bytes for Ed25519)
        signature = signature_path.read_bytes()

        # 2. Load public key
        pub_key = load_pem_public_key(public_key_path.read_bytes())
        if not isinstance(pub_key, Ed25519PublicKey):
            logger.error("Public key %s is not an Ed25519 key", public_key_path)
            return False

        # 3. Stream-hash the model file and verify signature
        hasher = hashlib.sha256()
        with model_path.open("rb") as fh:
            while chunk := fh.read(DEFAULT_CHUNK_SIZE):
                hasher.update(chunk)
        digest = hasher.digest()

        # Ed25519 can sign arbitrary messages – we sign the SHA-256 digest to
        # keep signature size fixed regardless of model size.
        pub_key.verify(signature, digest)
        logger.info("Model signature verified successfully: %s", model_path)
        return True
    except (InvalidSignature, ValueError) as exc:
        logger.warning("Invalid model signature for %s: %s", model_path, exc)
        return False
    except Exception as exc:  # noqa: BLE001 – broad catch to trap any I/O issues
        logger.exception("Unexpected error during signature verification: %s", exc)
        return False


def calculate_sha256(model_path: Path | str) -> str:
    """Utility helper to calculate a SHA-256 hex digest of *model_path*."""
    model_path = Path(model_path)
    hasher = hashlib.sha256()
    with model_path.open("rb") as fh:
        while chunk := fh.read(DEFAULT_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()
