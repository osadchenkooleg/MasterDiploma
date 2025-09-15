from app.infrastructure.db.duckdb.repos import CodesRepo, EmbeddingsRepo, LanguagesRepo
from app.infrastructure.embedings.model_codebert import CodeEmbeddingModel

_lang_repo = LanguagesRepo()
_codes_repo = CodesRepo()
_emb_repo = EmbeddingsRepo()
_embed_model = None


def get_languages_repo():
    return _lang_repo


def get_codes_repo():
    return _codes_repo


def get_embeddings_repo():
    return _emb_repo


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = CodeEmbeddingModel()
    return _embed_model
