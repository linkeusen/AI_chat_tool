from dataclasses import dataclass
from enum import auto,Enum
import streamlit as st 
import torch
import os
from transformers import AutoTokenizer,AutoConfig,AutoModel
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token

# 定义Role,不同角色
# get_message()获取角色对应的消息
class Role(Enum):
    SYSTEM=auto()
    USER=auto()
    ASSISTANT=auto()
    TOOL=auto()
    INTERPRETER=auto()
    OBSERVATION=auto()
    # print时显示
    def __str__(self):
        match self:
            case Role.SYSTEM:
                return "<|system|>"
            case Role.USER:
                return "<|user|>"
            case Role.ASSISTANT | Role.TOOL | Role.INTERPRETER:
                return "<|assistant|>"
            case Role.OBSERVATION:
                return "<|observation|>"
    # 实例化后调用
    def get_message(self):
        match self.value:
            case Role.SYSTEM.value:
                return
            case Role.USER.value:
                return st.chat_message(name="user", avatar="user")
            case Role.ASSISTANT.value:
                return st.chat_message(name="assistant", avatar="assistant")
            case Role.TOOL.value:
                return st.chat_message(name="tool", avatar="assistant")
            case Role.INTERPRETER.value:
                return st.chat_message(name="interpreter", avatar="assistant")
            case Role.OBSERVATION.value:
                return st.chat_message(name="observation", avatar="user")
            case _:
                st.error(f'Unexpected role: {self}')

# conversation传入角色，内容，工具，图片，
#   get_text()获取处理后的文本，
#   Role.show()在网页显示聊天
@dataclass
class Conversation:
    role:Role
    content:str
    tool:str | None = None
    image:str | None = None

    def __str__(self):
        match self.role:
            case Role.SYSTEM | Role.USER | Role.ASSISTANT | Role.OBSERVATION:
                return f'{self.role}\n{self.content}'
            case Role.TOOL:
                return f'{self.role}{self.tool}\n{self.content}'
            case Role.INTERPRETER:
                return f'{self.role}interpreter\n{self.content}'
    def get_text(self):
        text=postprocess_text(self.content)
        match self.role:
            case Role.TOOL:
                text=f'Calling tool `{self.tool}`:\n\n{text}'
            case Role.INTERPRETER.value:
                text=f'{text}'
            case Role.OBSERVATION.value:
                text=f'Observation:\n```\n{text}\n```'
        return text
    def show(self,placeholder=None):
        if placeholder:
            message=placeholder
        else:
            message=self.role.get_message()
        if self.image:
            message.image(self.image)
        else:
            text=self.get_text()
            message.markdown(text)

# 处理文本的
def postprocess_text(text):
    text = text.replace("\(", "$")
    text = text.replace("\)", "$")
    text = text.replace("\[", "$$")
    text = text.replace("\]", "$$")
    text = text.replace("<|assistant|>", "")
    text = text.replace("<|observation|>", "")
    text = text.replace("<|system|>", "")
    text = text.replace("<|user|>", "")
    return text.strip()

# 传入conversation这个类和history历史
def append_conversation(conversation,history,placeholder=None):
    history.append(conversation)
    conversation.show(placeholder)

#
TOOL_PROMPT = 'Answer the following questions as best as you can. You have access to the following tools:'
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
PT_PATH = os.environ.get('PT_PATH', None)
PRE_SEQ_LEN = int(os.environ.get("PRE_SEQ_LEN", 128))

# 聊天输出函数,输入参数->model，tokenizer，history，role，其他各类参数
# 返回response,new_history,past_key_values
def stream_chat(model,tokenizer,query,history=None,role="user",
                logits_processor=None,
                past_key_values=None,
                max_new_tokens: int = 256,
                do_sample=True, 
                top_p=0.8,
                temperature=0.8,
                repetition_penalty=1.0,
                length_penalty=1.0, 
                num_beams=1,
                return_past_key_values=False,
                **kwargs
                ):
    class InvalidScoreLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                scores.zero_()
                scores[..., 5] = 5e4
            return scores
    if history is None:
        history=[]
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    eos_token_id =[tokenizer.eos_token_id,tokenizer.get_command("<|user|>"),tokenizer.get_command("<|observation|>")]
    gen_kwargs={"max_new_tokens": max_new_tokens,
                  "do_sample": do_sample,
                  "top_p": top_p,
                  "temperature": temperature,
                  "logits_processor": logits_processor,
                  "repetition_penalty": repetition_penalty,
                  "length_penalty": length_penalty,
                  "num_beams": num_beams,
                  **kwargs
                  }
    if past_key_values is None:
        # inputs={'input_ids': tensor([[64790, 64792,..]]), 'attention_mask': tensor([[1, 1, ..]), 'position_ids': tensor([[0, 1, ..]])}
        # decode:[gMASK]sop<|user|> nihao<|assistant|>
        inputs=tokenizer.build_chat_input(query,history=history,role=role)
    else:
        inputs=tokenizer.build_chat_input(query,role=role)
        past_length=past_key_values[0][0].shape[0]
        if model.transformer.pre_seq_len is not None:
            past_length-=model.transformer.pre_seq_len
        inputs.position_ids+=past_length
        attention_mask=inputs.attention_mask
        attention_mask=torch.cat((attention_mask.new_ones(1,past_length), attention_mask),dim=1)
        inputs['attention_mask']=attention_mask
    inputs=inputs.to(model.device)
    history.append({"roel":role,"content":query})
    input_sequence_length=inputs["input_ids"].shape[1]
    if input_sequence_length+max_new_tokens>model>=model.config.seq_length:
        yield "当前输入序列长度{}加上新token的最大长度{},超过最大序列长度{}".format(input_sequence_length,max_new_tokens,model.config.seq_length),history
        return
    if input_sequence_length > model.config.seq_length:
        yield "当前输入序列长度{}，超过最大序列长度{}".format(input_sequence_length,model.config.seq_length),history
        return
    for outputs in model.stream_generate(**inputs,
                                         past_key_values=past_key_values,
                                         eos_token_id=eos_token_id,
                                         return_past_key_values=return_past_key_values,
                                         **gen_kwargs):
        if return_past_key_values:
            outputs,past_key_values=outputs
        outputs=outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response=tokenizer.decode(outputs)
        if response and response[-1] !="�"
            new_history=history
            if return_past_key_values:
                yield response,new_history,past_key_values
            else:
                yield response,new_history

# 这个类初始化需要model_path,tokenizer_path,pt_checkpoint
#   generate_steam(需要system，tools，history，和各类其他参数)返回TextGenerationStreamResponse对象
class Client:
    def __init__(self,model_path,tokenizer_path,pt_checkpoint):
        self.model_path=model_path
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)
        if pt_checkpoint is not None and os.path.exists(pt_checkpoint):
            config=AutoConfig.from_pretrained(model_path,trust_remote_code=True,pre_seq_len=PRE_SEQ_LEN)
            self.model=AutoModel.from_pretrained(model_path,trust_remote_code=True,config=config,device_map="auto").eval()
            # add .quantize(bits=4, device="cuda").cuda() before .eval() and remove device_map="auto" to use int4 model
            # must use cuda to load int4 model
            prefix_state_dict=torch.load(os.path.join(pt_checkpoint,"pytorch_model.bin"))
            new_prefix_state_dict={}
            for k,v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            self.model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        else:
            self.model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

    def generate_stream(self,system:str,tools:list[dict],history,**parameters):
        chat_history = [{'role':'system','content':system if not tools else TOOL_PROMPT}]
        if tools:
            chat_history[0]['tools']=tools
        for conversation in history[-1]:
            chat_history.append({'role':str(conversation.role).removeprefix('<|').removesuffix('|>'),'content':conversation.content})
        query=history[-1].content
        role=str(history[-1].role).removeprefix('<|').removesuffix('|>')
        text=''
        for response,_ in stream_chat(self.model,self.tokenizer,query,chat_history,role,**parameters):
            word=response.removeprefix('text')
            word_stripped=word.strip()
            text=response
            yield TextGenerationStreamResponse(
                generated_text=text,
                token=Token(
                    id=0,
                    logprob=0,
                    text=word,
                    special=word_stripped.startswith('<|') and word_stripped.endswith('|>'),
                )
            )

# 实例化client
@st.cache_resouce
def get_client():
    client=Client(MODEL_PATH,TOKENIZER_PATH,PT_PATH)
    return client


# chat主函数，传入prompt,system_prompt,client,其他参数模型生成参数
#   输出
def chat_main(prompt_text,system_prompt,client,top_p=0.8,temperature=0.95,repetition_penalty=1.0,max_new_tokens=1024,retry=False):
    placeholder=st.empty()
    with placeholder.container():
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history=[]
    if prompt_text == '' and retry==False:
        print("\n== Clean ==\n")
        st.session_state.chat_history=[]
        return
    history=st.session_state.chat_history
    for conversation in history:
        conversation.show()
    if retry:
        print("=============retry=============")
        last_user_conversation_idx=None
        for idx,conversation in enumerate(history):
            if conversation.role==Role.USER:
                last_user_conversation_idx=idx 
            if last_user_conversation_idx is not None:
                prompt_text=history[last_user_conversation_idx].content
                del history[last_user_conversation_idx:]
    if prompt_text:
        prompt_text = prompt_text.strip()
        append_conversation(Conversation(Role.USER, prompt_text), history)
        placeholder = st.empty()
        message_placeholder = placeholder.chat_message(name="assistant", avatar="assistant")
        markdown_placeholder = message_placeholder.empty()

        output_text = ''
        for response in client.generate_stream(
                system_prompt,
                tools=None,
                history=history,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=[str(Role.USER)],
                repetition_penalty=repetition_penalty,
        ):
            # response->{generated_text,token,special}
            token = response.token
            # special=word_stripped.startswith('<|') and word_stripped.endswith('|>')
            if response.token.special:
                print("\n==Output:==\n", output_text)
                match token.text.strip():
                    case '<|user|>':
                        break
                    case _:
                        st.error(f'Unexpected special token: {token.text.strip()}')
                        break
            output_text += response.token.text
            markdown_placeholder.markdown(postprocess_text(output_text + '▌'))

        append_conversation(Conversation(Role.ASSISTANT,postprocess_text(output_text),), history, markdown_placeholder)

# 工具主函数
def tool_main():
    return

# 代码解释器主函数
def ci_main():
    return

if __name__ =='__main__':

    DEFAULT_SYSTEM_PROMPT = '''
    You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
    '''.strip()

    # 页面配置
    st.set_page_config(
        page_title="Chat&tool",
        page_icon=":robot:",
        layout='centered',
        initial_sidebar_state='expanded',
    )

    st.title("Chat&tool")
    st.markdown("选择不同模式")

    class Mode(Enum):
        CHAT,TOOL,CI = '💬 Chat', '🛠️ Tool', '🧑‍💻 Code Interpreter'

    # 侧边栏
    with st.sidebar:
        top_p = st.slider(
            'top_p', 0.0, 1.0, 0.8, step=0.01
        )
        temperature = st.slider(
            'temperature', 0.0, 1.5, 0.95, step=0.01
        )
        repetition_penalty = st.slider(
            'repetition_penalty', 0.0, 2.0, 1.1, step=0.01
        )
        max_new_token = st.slider(
            'Output length', 5, 32000, 256, step=1
        )

        cols = st.columns(2)
        export_btn = cols[0]
        clear_history = cols[1].button("Clear History", use_container_width=True)
        retry = export_btn.button("Retry", use_container_width=True)

        system_prompt = st.text_area(
            label="System Prompt (Only for chat mode)",
            height=300,
            value=DEFAULT_SYSTEM_PROMPT,
        )

    # 输入prompt
    prompt_text =st.chat_input('input:',key='chat_input')
    if clear_history or retry:
        prompt_text=""

    # 选择模式
    client = get_client()
    tab=st.radio([mode.value for mode in Mode],horizontal=True)
    match tab:
        case Mode.CHAT:
            chat_main(prompt_text,
                    system_prompt,
                    client,
                    top_p,
                    temperature,
                    repetition_penalty
                    ,max_new_token,retry)
        case Mode.TOOL:
            tool_main(

            )
        case Mode.CI:
            ci_main(

            )
            



