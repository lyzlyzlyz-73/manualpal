import json

from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.import_process.nodes.entry import EntryNode
from knowledge.processor.import_process.nodes.pdf_to_md import PdfToMdNode
from knowledge.processor.import_process.nodes.md_img import MdImgNode
from knowledge.processor.import_process.nodes.ducment_split import DocumentSplitNode
from knowledge.processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.bge_embedding import BgeEmbeddingNode
from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode
from knowledge.processor.import_process.state import ImportGraphState

def router(state:ImportGraphState)->str:
    if state.get("is_pdf_read_enabled"):
        return "pdf"
    if state.get("is_md_read_enabled"):
        return "md"
    return "END"


def create_import_graph()->CompiledStateGraph:
    graph=StateGraph(ImportGraphState)
    nodes={
        "entry_node":EntryNode(),
        "pdf_to_md_node":PdfToMdNode(),
        "md_img_node":MdImgNode(),
        "document_split_node":DocumentSplitNode(),
        "item_name_recognition_node":ItemNameRecognitionNode(),
        "bge_embedding_node":BgeEmbeddingNode(),
        "import_milvus_node":ImportMilvusNode(),
    }
    nb=1#set_entry_point只传节点名
    for name,node in nodes.items():
        graph.add_node(name,node)
    graph.set_entry_point("entry_node") 
    nb=1#router会自动添加参数
    graph.add_conditional_edges(
        "entry_node",
        router,
        {
            "pdf":"pdf_to_md_node",
            "md":"md_img_node",
            "END":END
        }
    )

    graph.add_edge("pdf_to_md_node","md_img_node")
    graph.add_edge("md_img_node","document_split_node")
    graph.add_edge("document_split_node","item_name_recognition_node")
    graph.add_edge("item_name_recognition_node","bge_embedding_node")
    graph.add_edge("bge_embedding_node","import_milvus_node")
    graph.add_edge("import_milvus_node",END)
    return graph.compile()

kb_import_graph=create_import_graph()

def run_import_graph(import_file_path:str,file_dir:str)->dict:
    state={
        "import_file_path":import_file_path,
        "file_dir":file_dir,
        "is_md_read_enabled":False,
        "is_pdf_read_enabled":True
    }
    finally_state={}
    nb=250#stream流式调用，节点执行完->{name：state}
    for event in kb_import_graph.stream(state):
        for name_node,state in event.items():
            print(name_node)
            finally_state=state
    return finally_state

nb=250#main函数只有运行此文件执行，顶层代码块import，运行此文件都执行
if __name__== "__main__":
    import_file_path="G:\\LLL\LLL\\2.资料\\pdf文档\\doc\\万用表RS-12的使用.pdf"
    file_dir="G:\\LLL\\LLL\\2.资料\\pdf文档\\test_runimportgraph"
    finally_state=run_import_graph(import_file_path,file_dir)
    nb=250#json.dumps()将字典转换为json字符串，indent=4缩进4个空格，ensure_ascii=False允许中文字符
    print(json.dumps(finally_state, indent=4,ensure_ascii=False))
    kb_import_graph.get_graph().print_ascii()
