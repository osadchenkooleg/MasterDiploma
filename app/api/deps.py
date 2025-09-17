# from app.infrastructure.db.clickhouse.codes_repo import CodesRepoCH
# from app.infrastructure.db.clickhouse.embeddings_repo import EmbeddingsRepoCH
# from app.infrastructure.db.clickhouse.languages_repo import LanguagesRepoCH
from app.infrastructure.embeddings.model_codebert import CodeEmbeddingModel


#
# _lang_repo = LanguagesRepoCH()
# _codes_repo = CodesRepoCH()
# _emb_repo = EmbeddingsRepoCH()
# _embed_model = None
#
#
# def get_languages_repo():
#     return _lang_repo
#
#
# def get_codes_repo():
#     return _codes_repo
#
#
# def get_embeddings_repo():
#     return _emb_repo
#
#
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = CodeEmbeddingModel()
    return _embed_model
