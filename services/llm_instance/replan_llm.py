from services.llm import LLMFactory

llm = None


def get_instance():
    global llm
    if llm is None:
        llm = LLMFactory.create()
    return llm