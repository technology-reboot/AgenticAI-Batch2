import argparse
import json
import os
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERSIST_DIR = Path("faiss_index")
COLLECTION_NAME = "it_companies"
CORPUS_DIR = Path("data/company_profiles")


CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 5
THRESHOLDS =[0.5, 0.7, 0.8]
REFUSAL = " I do not have that information"

PROFILES = {
    "tcs.txt": (
        "TCS",
        "Tata Consultancy Services",
        "Tata Consultancy Services (TCS) was founded in 1968 and is headquartered in Mumbai. "
        "It is part of the Tata Group, and K Krithivasan became chief executive officer in June 2023. "
        "TCS employs approximately 600,000 people and is India's largest IT services exporter.\n\n"
        "TCS helps organizations modernize applications, manage infrastructure, build cloud platforms, "
        "and use data and artificial intelligence responsibly. Its work spans banking, insurance, retail, "
        "manufacturing, telecommunications, healthcare, travel, and the public sector. It combines "
        "consulting, implementation, and managed services for large enterprises.\n\n"
        "TCS invests in employee training, cloud-native architecture, automation, and industry software. "
        "Its global delivery network supports customers across regions and time zones. This profile is a "
        "stable training source, not live market data. It contains no stock prices or sports results."
    ),
    "infosys.txt": (
        "Infosys",
        "Infosys",
        "Infosys was founded in 1981 in Pune by N. R. Narayana Murthy and six others. Its headquarters "
        "are now in Bengaluru, and Salil Parekh is its chief executive officer. For FY2025, Infosys "
        "provided revenue growth guidance of 3-4% in constant currency.\n\n"
        "Infosys provides consulting, technology implementation, application management, cloud services, "
        "cybersecurity, data analytics, and business process services. Its customers work in financial "
        "services, retail, communications, manufacturing, healthcare, and energy. Distributed teams "
        "support customers in North America, Europe, Asia Pacific, and other markets.\n\n"
        "Infosys emphasizes continuous learning, cloud adoption, automation, artificial intelligence, "
        "and responsible data use. This is a bounded training profile with facts as of FY2025, not a live "
        "information service. It contains no stock prices, market capitalization, or sports results."
    ),
    "wipro.txt": (
        "Wipro",
        "Wipro",
        "Wipro was founded in 1945 as Western India Vegetable Products and later pivoted to information "
        "technology in the 1980s. It is headquartered in Bengaluru. Srini Pallia was appointed chief "
        "executive officer in 2024. Wipro serves clients in 66 countries.\n\n"
        "Wipro works across consulting, applications, cloud, infrastructure, cybersecurity, engineering, "
        "data, and business process services. Its customers include organizations in banking, healthcare, "
        "consumer products, communications, energy, manufacturing, and public services.\n\n"
        "Wipro's international delivery network combines Indian engineering centers with regional teams. "
        "It invests in technical learning, energy efficiency, and community programs. This profile is a "
        "bounded training document; it contains no stock prices, market capitalization, or sports results."
    ),
    "hcltech.txt": (
        "HCLTech",
        "HCLTech",
        "HCLTech was founded in 1976 and is headquartered in Noida. C Vijayakumar is its chief executive "
        "officer. The company has a strong engineering-and-R&D services mix, supporting organizations "
        "that design, build, modernize, and operate technology products and enterprise systems.\n\n"
        "Its services include digital engineering, product development, cloud transformation, applications, "
        "infrastructure, cybersecurity, data, and artificial intelligence. HCLTech serves clients in "
        "technology, financial services, manufacturing, healthcare, life sciences, and retail.\n\n"
        "HCLTech connects product engineering with enterprise operations through global teams and repeatable "
        "quality practices. This is a training corpus profile, not live market data. It contains no stock "
        "prices, market capitalization, or sports results."
    ),
    "tech_mahindra.txt": (
        "TechMahindra",
        "Tech Mahindra",
        "Tech Mahindra was founded in 1986 and is headquartered in Pune. It is part of the Mahindra Group. "
        "Mohit Joshi became chief executive officer in December 2023. The company has a telecom-heavy "
        "client base and serves communications providers as well as customers in other industries.\n\n"
        "Its capabilities include network services, engineering, cloud, applications, cybersecurity, "
        "customer experience, data, and business process services. Telecom programs include network "
        "transformation, operations support, service assurance, and customer care.\n\n"
        "Tech Mahindra combines consulting and engineering with implementation and managed services. "
        "This is a bounded training profile with facts as of FY2025. It contains no stock prices, market "
        "capitalization, or sports results."
    ),
}

DEFAULT_EVAL = [
    {"id": "q01", "question": "Who is the CEO of TCS?", "gold_companies": ["TCS"], "type": "single"},
    {"id": "q02", "question": "When was Infosys founded and by whom?", "gold_companies": ["Infosys"], "type": "single"},
    {"id": "q03", "question": "What is Infosys revenue growth guidance for FY2025?", "gold_companies": ["Infosys"], "type": "single"},
    {"id": "q04", "question": "In how many countries does Wipro serve clients?", "gold_companies": ["Wipro"], "type": "single"},
    {"id": "q05", "question": "Where is HCLTech headquartered?", "gold_companies": ["HCLTech"], "type": "single"},
    {"id": "q06", "question": "Which group does Tech Mahindra belong to?", "gold_companies": ["TechMahindra"], "type": "single"},
    {"id": "q07", "question": "Compare the founding years of TCS and Infosys", "gold_companies": ["TCS", "Infosys"], "type": "multi_hop"},
    {"id": "q08", "question": "Which of these companies is headquartered in Pune?", "gold_companies": ["TechMahindra", "Infosys"], "type": "multi_hop"},
    {"id": "q09", "question": "Which company started as a vegetable products business?", "gold_companies": ["Wipro"], "type": "single"},
    {"id": "q10", "question": "Who won the FIFA World Cup in 2022?", "gold_companies": [], "type": "out_of_domain"},
]

GROUNDED_PROMT = ChatPromptTemplate.from_template(
'''Answer the question using ONLY the information in the context below
if the answer is not in the context, replay exactly:
"I do not have that information."
Answer in 2-3 sentences. Start by directly answering the question.
After your answer, add a line beginning "Source:" citing the context used. 

Context:{context}

Qeustion: {question}

Asnwer: 
'''
)

UNGROUNDED_PROMT = ChatPromptTemplate.from_template(
'''Use the context below to help you answer the question
Answer in 2-3 sentences. Start by directly answering the question.
After your answer, add a line beginning "Source:" citing the context used. 

Context:{context}

Qeustion: {question}

Asnwer: 
'''
)

def require_api_key():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI Key is missing")

def build_corpus() -> int:
    CORPUS_DIR.mkdir(parents=tRUE, exist_ok=True)
    written = 0
    for filename, (code, comapny_name, body) in PROFILES.items():
        path = CORPUS_DIR / filename
        if not path.exists():
            path.write_text(
                f"{comapny_name} company profile (training corpus, facts as of FY2025)\n\n{body}]n", 
                encoding="utf-8"
            )
            written+=1
        return written

def load_documents():
    company_by_filename = {filename: code for filename, (code, _, _) in PROFILES.items()}
    documents = []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        document = TextLoader(str(path), encoding="urf-8").load()[0]
        document.metadata.update(
            company=company_by_filename.get(path.name, path.stem),
            source=path.name,
            doc_type="profile"
        )
        documents.append(document)
    if not document:
        raise SystemExit("No files found")
    return documents


