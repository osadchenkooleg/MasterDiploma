# from clickhouse_connect import get_client
#
# from app.infrastructure.embeddings.model_codebert import (  # or .infra.
#     CodeEmbeddingModel,
# )
#
# m = CodeEmbeddingModel()  # prints device
# ch = get_client(
#     host="localhost", username="default", password="1234", database="codebase"
# )
#
#
# def top5(code: str, langs=None):
#     qvec = m.encode(code).tolist()
#     params = {"qvec": qvec, "langs": langs or []}
#     rows = ch.query(
#         """
#       SELECT code_id, 1 - cosineDistance(vector, {qvec:Array(Float32)}) AS sim
#       FROM codebase.embeddings_v2
#       WHERE (empty({langs:Array(String)}) OR lang IN {langs:Array(String)})
#         AND transform_ver = 2
#       ORDER BY cosineDistance(vector, {qvec:Array(Float32)}) ASC, code_id ASC
#       LIMIT 5
#     """,
#         parameters=params,
#     ).result_rows
#     return rows
#
#
# print(top5("public class A { int x; }", ["java"]))
# print(top5("def add(a,b): return a+b", ["python"]))
