import time
import os
import copy
from typing import Iterable, List
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.schema.vectorstore import VectorStore
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.embeddings import Embeddings

class ExecutionTimeChecker:
  def __init__(self, with_start: bool = False):
    """
    ExecutionTimeChecker 클래스의 생성자입니다.

    :param with_start: 시간 측정을 시작할지 여부를 결정하는 부울 값
    """
    self.start_time = 0
    self.end_time = 0
    if with_start:
      self.start_time = time.time()

  def start(self) -> 'ExecutionTimeChecker':
    """
    시간 측정을 시작하고, self를 반환합니다.

    :return: self (ExecutionTimeChecker 객체)
    """
    self.start_time = time.time()
    self.end_time = 0
    return self

  def stop(self) -> float:
    """
    시간 측정을 종료하고 경과 시간을 반환합니다.

    :return: 경과 시간 (초 단위)
    """
    self.end_time = time.time()
    return self.get_elapsed_time()

  def get_elapsed_time(self) -> float:
    """
    시간 측정 결과인 경과 시간을 반환합니다.

    :return: 경과 시간 (초 단위)
    """
    if self.start_time:
      current_time = time.time()
      if self.end_time:
        return self.end_time - self.start_time
      else:
        return current_time - self.start_time
    else:
      return None



class ConfigManager:
  default_config = None
  def __init__(self, config = {}):
    """
    ConfigManager 클래스의 생성자입니다. 이 클래스는 설정 항목을 관리합니다.

    :param config: 초기 설정 항목을 담긴 딕셔너리 또는 기본값 ( 설정되지 않은 경우 )
    """
    self.config = copy.deepcopy(config)

  def set(self, key, value):
    """
    설정 항목을 설정합니다.

    :param key: 설정 항목의 키
    :param value: 설정 항목의 값
    """
    self.config[key] = value

  def get(self, key, default=None):
    """
    설정 항목의 값을 가져옵니다.

    :param key: 설정 항목의 키
    :param default: 설정 항목이 없을 경우 반환할 기본값 (선택 사항)
    :return: 설정 항목의 값 또는 기본값 (설정되지 않은 경우)
    """
    return self.config.get(key, default)

  def get_all(self):
    """
    모든 설정 항목을 가져옵니다.

    :return: 모든 설정 항목을 담은 딕셔너리
    """
    return self.config

  @classmethod
  def get_env(cls, key, default=None):
    """
    환경 변수에서 설정 항목의 값을 가져옵니다.

    :param key: 설정 항목의 키
    :param default: 설정 항목이 없을 경우 반환할 기본값 (선택 사항)
    :return: 환경 변수에서 가져온 설정 항목의 값 또는 기본값 (설정되지 않은 경우)
    """
    return os.environ.get(key, default)

  @classmethod
  def get_default_config(cls, config = {}):
    """
    클래스의 기본 설정(ConfigManager 객체)을 가져옵니다. 기본 설정은 클래스 레벨에서 공유되며,
    필요한 경우 주어진 구성(config)으로 초기화됩니다.

    :param config: 초기 설정을 나타내는 딕셔너리 (선택 사항)
    :return: ConfigManager 객체
    """
    if not cls.default_config:
      cls.default_config = ConfigManager(config)

    return cls.default_config


def color_code(color: str) -> str:
  """
  입력한 색상 이름을 콘솔 텍스트 색상을 나타내는 ANSI 이스케이프 코드로 변환합니다.

  매개변수:
    color (str): 변환할 색상 이름.

  반환:
    str: 지정된 색상에 해당하는 ANSI 이스케이프 코드 또는 해당 색상이 없는 경우 빈 문자열.

  예시:
    초록색 텍스트 출력하기:
    print(color_code("green") + "이것은 초록색 텍스트입니다." + color_code("reset"))
  """
  color_map = {
    "reset": "\033[0m",
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m"
  }
  return color_map.get(color.lower(), "")



class ConsoleInput:
  def __init__(self, enter_blank_line=0, basic_prompt='', prompt_color="bright_blue", user_color="yellow", answer_color="green"):
    """
    ConsoleInput 클래스의 생성자입니다.

    Args:
      enter_blank_line (int): 연속된 빈 줄의 입력 횟수를 설정합니다.
      basic_prompt (str): 기본 프롬프트로 사용할 텍스트입니다.
      prompt_color (str): 프롬프트의 색상을 지정합니다.
      user_color (str): 사용자 입력의 색상을 지정합니다.
      answer_color (str): 출력할 답변 텍스트의 색상을 지정합니다.
    """
    self.enter_blank_line = enter_blank_line
    self.basic_prompt = basic_prompt
    self.prompt_color = prompt_color
    self.user_color = user_color
    self.answer_color = answer_color

  def input(self, prompt='') -> List[str]:
    """
    사용자로부터 여러 줄의 입력을 받아 리스트로 반환합니다.

    Args:
      prompt (str): 사용자에게 표시할 프롬프트 텍스트입니다.

    Returns:
      List[str]: 사용자가 입력한 여러 줄의 텍스트를 담은 리스트입니다.
    """
    blank_prompt = ' ' * len(self.basic_prompt)
    lines = []  # 사용자로부터 입력받은 문장을 저장할 리스트
    blank_count = 0  # 연속으로 입력받은 빈 줄의 수를 저장할 변수
    reset_color = color_code("reset")
    color_str = lambda color, text : color_code(color) + text + reset_color

    while True:
      _prompt = self.basic_prompt if blank_count == 0 else blank_prompt
      _prompt = color_code(self.prompt_color) + _prompt + prompt + color_code(self.user_color)

      user_input = input(_prompt)  # 사용자로부터 문장 입력 받기
      print(reset_color, end='')

      if self.enter_blank_line == 0:
        return [user_input]

      lines.append(user_input)  # 문장 저장

      if user_input == '':  # 입력받은 문장이 빈 줄이면
        blank_count += 1  # 빈 줄 카운트 증가
        if blank_count >= self.enter_blank_line:
          return lines[:-self.enter_blank_line]
      else:
        blank_count = 0  # 빈 줄이 아니면 카운트 리셋

  def out_str(self, answer: str) -> str:
    """
    지정된 답변 문자열을 콘솔에 출력하기 위한 ANSI 이스케이프 코드를 적용합니다.

    Args:
      answer (str): 출력할 답변 문자열.

    Returns:
      str: ANSI 이스케이프 코드가 적용된 답변 문자열.
    """
    return color_code(self.answer_color) + answer + color_code('reset')

  def out(self, answer: str) -> None:
    """
    지정된 답변 문자열을 콘솔에 출력합니다.

    Args:
      answer (str): 출력할 답변 문자열.

    Returns:
      None
    """
    print(self.out_str(answer))



import sys
import time
import threading


class BusyIndicator:
  def __init__(self, print_time = False, msg = ''):
    """
    BusyIndicator 클래스의 생성자입니다.

    생성된 인스턴스는 백그라운드에서 "busy" 상태를 반복 출력할 수 있습니다.
    """
    self.done_event = threading.Event()  # 스레드 종료 이벤트
    self.busy_thread = threading.Thread(target=self.run)  # 백그라운드 스레드
    self.busy_thread.daemon = True  # 백그라운드 스레드를 데몬으로 설정하여 메인 프로그램 종료 시 함께 종료됨
    self.start_time = time.time()
    self.print_time = print_time
    self.msg = msg

  def run(self):
    """
    "busy" 메시지를 반복 출력하고 백그라운드 스레드를 실행합니다.
    """
    while not self.done_event.is_set():
      time_str = ''
      for char in "-\|/":
        if self.print_time:
          time_str = self.msg + str(round(time.time() - self.start_time, 2)) + ' '
        sys.stdout.write(time_str + char )
        sys.stdout.flush()
        time.sleep(0.2)  # 문자가 변경되는 간격 (초 단위)

        sys.stdout.write("\033[2K")  # 현재 라인을 지우는 이스케이프 시퀀스
        sys.stdout.write("\033[G")   # 커서를 앞으로 이동하는 이스케이프 시퀀스

  def start(self):
    """
    백그라운드 스레드를 시작합니다.
    """
    self.start_time = time.time()
    self.busy_thread.start()  # 백그라운드 스레드 시작

  @classmethod
  def busy(cls, print_time = False, msg = ''):
    indicator = cls(print_time, msg)
    indicator.start()
    return indicator

  def stop(self):
    """
    백그라운드 스레드를 종료합니다.
    """
    self.done_event.set()
    self.busy_thread.join()






def get_filename_without_extension(file_path):
  """
  주어진 파일 경로에서 확장자를 제외한 파일명을 추출합니다.

  :param file_path: 파일 경로
  :return: 확장자를 제외한 파일명
  """
  # 파일 경로를 파싱하여 파일명과 확장자로 분리
  file_name, _ = os.path.splitext(os.path.basename(file_path))

  return file_name


def split_docs(docs: Iterable[Document], chunk_size=1500) -> List[Document]:
  """
  문서를 분할하는 함수입니다.

  :param docs: Document 객체의 Iterable
  :return: 분할된 Document 객체의 List
  """
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=150
  )

  splits = text_splitter.split_documents(docs)
  # print(f"len(splits)=>{len(splits)}")
  return splits


def load_pdf(file: str, with_split: bool = False, chunk_size=1500) -> List[Document]:
  """
  PDF 파일을 로드하는 함수입니다.

  :param file: PDF 파일의 경로
  :param with_split: 문서 분할 여부를 결정하는 부울 값
  :return: Document 객체의 List
  """
  loader = PyPDFLoader(file)
  pages = loader.load()
  if not with_split:
    return pages

  return split_docs(pages, chunk_size)


def load_csv(file: str, with_split: bool = False) -> List[Document]:
  """
  CSV 파일을 로드하는 함수입니다.

  :param file: PDF 파일의 경로
  :param with_split: 문서 분할 여부를 결정하는 부울 값
  :return: Document 객체의 List
  """
  loader = CSVLoader(file_path=file, encoding='utf-8')
  pages = loader.load()
  if not with_split:
    return pages

  return split_docs(pages)


def load_pdfs( files: Iterable[str], with_split: bool = False) -> List[Document]:
  """
  PDF 파일들을 로드하는 함수입니다.

  :param files: PDF 파일의 목록
  :param with_split: 문서 분할 여부를 결정하는 부울 값
  :return: Document 객체의 List
  """

  docs = []
  for file in files:
    docs.extend(load_pdf(file, with_split))

  return docs



def replace_korean_with_code(s: str) -> str:
    converted = ""
    for c in s:
        if c == ' ':  # 빈칸 확인
            converted += '_'
        elif '가' <= c <= '힣':  # 한글 범위 확인
            converted += str(ord(c))
        else:
            converted += c
    return converted


def get_vectordb_path(path1: str, path2: str = None) -> str:
  """
  벡터 데이터베이스(.vectordb) 파일의 경로를 반환합니다.

  :param path1: vector db가 위치할 상대 path 첫번째
  :param path1: vector db가 위치할 상대 path 두번째 ( 기본값은 비어 있음 )
  :return: 벡터 데이터베이스 폴더의 경로
  """
  if not path2 == None:
    path2 = replace_korean_with_code(path2)
  # 설정된 VECTORDBPATH 환경 변수 또는 기본 경로를 사용하여 벡터 데이터베이스 폴더 경로 생성
  path = os.path.join(ConfigManager.get_env("VECTORDBPATH", "./vectordb"), path1, path2)
  return path


def get_vectordb_path_by_file_path(file_path: str) -> str:
  """
  file path로 부터 벡터 데이터베이스(.vectordb) 파일의 경로를 반환합니다.
  <vectordb root>/<ext>/<filename>로 구성된다.

  :param file_path: 주어진 파일의 경로
  :return: 벡터 데이터베이스 폴더의 경로
  """

  file_name, file_extension = os.path.splitext(os.path.basename(file_path))
  return get_vectordb_path(file_extension[1:], file_name)


def get_pdf_vectordb_path(file: str) -> str:
  """
  PDF 벡터 데이터베이스(.vectordb) 파일의 경로를 반환합니다.

  :param file: PDF 파일의 경로 또는 파일명
  :return: PDF 벡터 데이터베이스 폴더의 경로
  """
  # 파일명에서 확장자를 제외한 부분 추출
  file_name = get_filename_without_extension(file)

  return get_vectordb_path("pdf", file_name)


def save_vectordb(embedding: Embeddings, db_path: str, documents: List[Document]) -> VectorStore:
  """
  벡터 데이터베이스를 저장하고 반환합니다.

  :param embedding: Embeddings 객체
  :param db_path: 벡터 데이터베이스 파일 경로
  :param documents: 저장할 문서 리스트
  :return: 저장된 VectorStore 객체
  """
  # 문서와 임베딩을 사용하여 벡터 데이터베이스 생성
  vectordb = FAISS.from_documents(
    documents=documents,
    embedding=embedding
  )

  # 로컬 파일로 벡터 데이터베이스 저장
  vectordb.save_local(db_path)

  return vectordb

def load_vectordb(embedding: Embeddings, db_path: str) -> VectorStore:
  """
  벡터 데이터베이스를 로드하고 반환합니다.

  :param embedding: Embeddings 객체
  :param db_path: 벡터 데이터베이스 파일 경로
  :return: 로드된 VectorStore 객체
  """
  # 로컬 파일로부터 벡터 데이터베이스 로드
  vectordb = FAISS.load_local(embeddings=embedding, folder_path=db_path)

  return vectordb

def load_pdf_vectordb(file: str) -> VectorStore:
  """
  PDF 벡터 데이터베이스를 로드하거나 생성하고 반환합니다.

  :param file: PDF 파일의 경로 또는 파일명
  :return: 로드된 또는 생성된 VectorStore 객체
  """
  # PDF 벡터 데이터베이스 파일 경로 가져오기
  db_path = get_pdf_vectordb_path(file)

  vectordb: VectorStore = None
  documents: List[Document] = None

  # 벡터 데이터베이스 파일이 존재하지 않으면 PDF를 로드하고 벡터 데이터베이스 생성
  if not os.path.exists(db_path):
    documents = load_pdf(file, True)
    vectordb = save_vectordb(OpenAIEmbeddings(), db_path, documents)
  else:
    # 벡터 데이터베이스 파일이 이미 존재하면 로드
    vectordb = load_vectordb(OpenAIEmbeddings(), db_path)

  return vectordb



def load_vectordb_from_file(file: str) -> VectorStore:
  db_path = get_vectordb_path_by_file_path(file)

  if os.path.exists(db_path):
     # 벡터 데이터베이스 파일이 이미 존재하면 로드
    vectordb = load_vectordb(OpenAIEmbeddings(), db_path)
    return vectordb

  _, ext = os.path.splitext(file)
  ext = ext[1:]
  documents: List[Document] = None

  if ext == "pdf":
    documents = load_pdf(file, True)
  elif ext == "csv":
    documents = load_csv(file, True)
  else:
    raise ValueError(f"지원하지 않는 파일 포맷입니다.:{os.path.basename(file)}")

  vectordb = save_vectordb(OpenAIEmbeddings(), db_path, documents)
  return vectordb



def save_text(file: str, text: str) -> bool:
  # Extract directory from the file path
  directory = os.path.dirname(file)

  # If the directory does not exist, create it
  if not os.path.exists(directory):
    try:
      os.makedirs(directory)
    except OSError as error:
      print(f"Error creating directory: {error}")
      return False

  # Try to save the text to the file
  try:
    with open(file, 'w', encoding='utf-8') as f:
      f.write(text)
    return True
  except IOError as error:
    print(f"Error writing to file: {error}")
    return False


def read_text(file: str) -> str:
  try:
    with open(file, 'r', encoding='utf-8') as f:
      return f.read()
  except IOError as error:
    print(f"Error reading file: {error}")
    return ""


def load_summary_from_file(llm, file: str, combine_template: str=None) -> str:
  db_path = get_vectordb_path_by_file_path(file)
  summary_file = db_path+"_summary.txt"

  if os.path.exists(summary_file):
    return read_text(summary_file)

  docs = load_pdf(file, 8000)

  template = '''다음의 내용을 6000자 이상의 한글로 요약해줘:

{text}
'''

  prompt = PromptTemplate(template=template, input_variables=['text'])
  combine_prompt = PromptTemplate(template=combine_template, input_variables=['text']) if combine_template is not None else None

  params = {
    "map_prompt" : prompt,
    "combine_prompt" :combine_prompt,
  }
  # Summarize the content using TextRank algorithm
  chain = load_summarize_chain(llm,
                              chain_type="map_reduce",
                              **{k: v for k, v in params.items() if v is not None}
                              )
  summary: str = chain.run(docs)



  if not isinstance(summary, str):
    raise ValueError("summary is not str")

  if not save_text(summary_file, summary):
    raise ValueError(f"'{summary_file}' can not save")
  return summary