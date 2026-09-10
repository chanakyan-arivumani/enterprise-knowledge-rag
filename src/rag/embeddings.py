import ollama


def generate_embeddings(chunk: str, model="qwen3-embedding:0.6b"):
    return ollama.embed(model=model, input=chunk)["embeddings"][0]


def convert_text_list_to_embed_list(text_list: list) -> list:
    result = []
    id = 1
    for text in text_list:
        result.append({"id": id, "text": text, "embedding": generate_embeddings(text)})
        id += 1
    return result


def get_cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must be of same length")
    dot_product = 0
    a_sqrd = 0
    b_sqrd = 0
    for i in range(len(a)):
        dot_product += a[i] * b[i]
        a_sqrd += a[i] ** 2
        b_sqrd += b[i] ** 2
    mag_a = math.sqrt(a_sqrd)
    mag_b = math.sqrt(b_sqrd)
    if mag_a == 0 or mag_b == 0:
        raise ValueError("vectors must be non zero")
    return dot_product / (mag_a * mag_b)
