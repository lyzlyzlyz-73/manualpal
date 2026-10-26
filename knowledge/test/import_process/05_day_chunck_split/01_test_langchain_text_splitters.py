#pip install -U langchain_text_splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "苹果的颜色是红色的[SEP]香蕉的颜色是黄色的[SEP]橘子的颜色是橙色的"
splitter_false = RecursiveCharacterTextSplitter(
    chunk_size=20,#每个块的最大长度：必须小于文本长度，否则整段原样返回不切分（关键坑！）
    chunk_overlap=0,#相邻块之间重叠的字符数
    keep_separator=False,#False:丢弃分隔符（切出的块里不含 [SEP]）
    separators=["[SEP]"]#分隔符必须写成列表；写字符串会被拆成 '[' 'S' 'E' 'P' ']' 逐个当分隔符
)
print(splitter_false.split_text(text))
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter_false=RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=0,
    keep_separator=False,
    separators=["[SEP]"]
)
print(splitter_false.split_text(text))



