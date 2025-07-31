import gradio as gr
from rag_system import *
'''
    前端展示页面
'''
# 用于模拟 RAG 回答（替换为你自己的回答逻辑）
def chat_rag(history, user_input):

    prompt = """
    你是一个知识严谨、表达清晰的AI助手,请根据以下提供的参考资料回答用户的问题。
       - 回答必须仅基于资料内容，不要依赖你的常识或已有知识；
       - 如果资料中找不到相关信息，请明确说明“资料中未提及”；
       - 回答应简洁准确，如有引用请尽可能贴近原文表达；
       - 不要杜撰，不要引入资料中未包含的信息。

    用户问题：{}

    参考资料：
    {}

    请基于以上资料作答：
        """

    # 混合检索
    rerank_docs = mixed_retrieval(user_input, 5, 5, 3)
    rerank_docs = '\n\n'.join(rerank_docs)

    # 拼接Prompt，调用模型生成答案
    prompt = prompt.format(query1[0], rerank_docs)
    response = load_model_ollama(f'/no_think{prompt}')

    history.append((user_input, response))

    return history, ""

with gr.Blocks(theme=gr.themes.Base(
    primary_hue="pink",
    secondary_hue="cyan",
    neutral_hue="gray",
    font="Arial"
)) as demo:

    gr.HTML("""
    <style>
    body {
        background-color: #0d1b2a;
        color: #0d1b2a;
    }
    </style>
    """)

    gr.Markdown("<h1 style='text-align: center; color: deepskyblue;'>🧠 RAG 问答聊天机器人</h1>")

    chatbot = gr.Chatbot(label="RAG 对话", bubble_full_width=False, show_copy_button=True)

    with gr.Column():
        user_input1 = gr.Textbox(
            show_label=False, placeholder="请输入你的问题，例如：这段文本讲了什么？", lines=1
        )
        # user_input2 = gr.Textbox(
        #     show_label=False, placeholder="请输入该问题中的关键词", lines=1
        # )
        submit_btn = gr.Button("发送 🚀", variant="secondary")

    clear_btn = gr.Button("清空聊天", variant="secondary")

    # submit_btn.click(chat_rag, inputs=[chatbot, user_input1, user_input2], outputs=[chatbot, user_input1, user_input2])
    submit_btn.click(chat_rag, inputs=[chatbot, user_input1], outputs=[chatbot, user_input1])
    # user_input.submit(chat_rag, inputs=[chatbot, user_input], outputs=[chatbot, user_input])
    clear_btn.click(lambda: [], None, chatbot)

if __name__ == '__main__':

    query1 = ['北京大毫的公司代码是多少?']
    query2 = '大毫公司'

    demo.launch()
