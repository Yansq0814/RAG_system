import json

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    api_key='sk-e17052a8d9884c1cb36d3846f84bfc99',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-max")

with open('./evalute/eval_simple1.json', 'r', encoding='utf-8') as f:
    dataset = json.loads(f.read())


from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)

evaluator_llm = LangchainLLMWrapper(llm)
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
# result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()])
print(result)


