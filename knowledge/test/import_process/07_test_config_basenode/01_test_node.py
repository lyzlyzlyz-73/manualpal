import logging
import os
from abc import  ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Optional

T=TypeVar("T")

@dataclass
class Config():
    max_len_str:int=10
    min_len_str:int=field(default_factory=lambda:int(os.getenv("MIN_LEN_STR",5)))

    @classmethod
    def from_env(cls):
        return cls()

_config_test=None
def get_config_test():
    global _config_test
    if not _config_test:
        _config_test=Config().from_env()
        return _config_test
    return _config_test


class BaseNodeTest(ABC):
    name:str="test"

    def __init__(self,config_test:Optional[Config]=None):
        self.config_test=config_test or get_config_test()
        self.logger=logging.getLogger(f"import.{self.name}")

    def __call__(self,state:T)->T:
        task_id=state.get("task_id","default")
        try:
            self.logger.info(f"task {task_id}\nnode_name:{self.name}开始执行...")
            result=self.process(state)
            self.logger.info(f"task {task_id}\nnode_name:{self.name}执行完成...")
            return result
        except Exception as e:
            self.logger.error(f"task {task_id}\nnode_name:{self.name}执行出错...",e)
            raise e

    @abstractmethod
    def process(self,state:T)->T:
        pass




"""1. logger：父类 __init__ 里 self.logger = logging.getLogger(f"import.{self.name}")，
   子类只需定义 name，就自动获得专属 logger，直接用 self.logger.info(...) 记录。
   （每个节点 logger 名字不同，方便定位日志来源）

2. 属性存储：每个【实例】自带一个 __dict__（初始为空 dict）。
   @dataclass 生成的 __init__ 会执行 self.字段 = 值，把字段写进【实例的】__dict__；
   值的来源有两种——直接赋的常量，或 field(default_factory=...) 被调用后的返回值。
   普通类不做这件事，所以实例 __dict__ 是空的，读属性时从【类字典】里找。
   （类的 __dict__ 是 mappingproxy 只读视图，实例的 __dict__ 是普通 dict）

3. get_config()：返回【全局唯一的同一个】Config 实例（单例，第二次起吃缓存），
   改它会【全局生效】；只有手动 Config() 新建的对象才互不影响。
   单例的价值：调用极廉价（0.085 微秒/次），换来"不用到处传配置对象"。

4. TypeVar("T")：类型占位符，表达"多处同型"。具体类型由【类型检查器】推导，
   也可在 TypeVar 里加约束（枚举 int, float / 上界 bound=BaseClass）。
   注意：它【只在静态检查时有意义】，运行时完全不参与。
"""




