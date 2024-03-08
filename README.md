# Juliet-chat

# Project Overview

Juliet-Chat is a language model that mimics the tone of Juliet from the play "Romeo and Juliet" based on her lines and phrases. It has been fine-tuned using QLoRA with InternLM. Users can engage in conversations with Juliet-Chat through this model.

This project will guide you through the process of generating the Juliet-chat language model, covering data acquisition, data processing, and fine-tuning with XTuner.

# Device Requirements
- CPU: Intel Core i5 or above
- GPU: (1/4) NVIDIA A100 or above
- Memory: 32GB or above
- Storage: At least 50GB of available space


# Environment Configuration

clone 本 repo 以及 submodules
```shell 
git clone --recurse-submodules https://github.com/Linsy0107/Juilet-chat.git
```

<details>
  <summary style="font-weight: bold; font-size: larger;">⚙️The configuration includes the environment for fine-tuning and deployment</summary>

Create a New Environment - Install lmdeploy

Install LMDeploy using pip (for Python 3.8+), or [install from source](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/build.md)

```shell
conda create -n chatXY python=3.10 -y
pip install lmdeploy
```

The precompiled package of LMDeploy is compiled based on CUDA 11.8 by default. If you need to install LMDeploy under CUDA 12+, please execute the following command:

```shell
export LMDEPLOY_VERSION=0.2.0
export PYTHON_VERSION=38
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl
#比如pip install https://github.com/InternLM/lmdeploy/releases/download/v0.2.3/lmdeploy-0.2.3-cp310-cp310-manylinux2014_x86_64.whl
```

Install XTuner.
```shell
cd train/Xtuner
pip install -e '.[all]'
```

Install other dependencies.
```
pip install -r requirements.txt
```
</details>

# Data Acquisition

<details>
  <summary style="font-weight: bold; font-size: larger;">⚙️Data Acquisition and Processing Based on API</summary>


## precondition

1. OpenAI-format API
2. Python environment (refer to the environment configuration section )

## Data Composition 

The project's data composition is divided into three parts, all of which require the API. Choosing any two will yield satisfactory results.

- Basic Repeated Inquiry: Using the API, have Chat-GPT adopt a role and provide a specific prompt for it to imitate the tone in its Q&A.
- Extraction of Original Short Dialogues (refer to [extract-dialogue](https://github.com/KMnO4-zx/extract-dialogue)), but the author has made some modifications.
- Extraction of Original Long Dialogues

## Data Acquisition

### 1.Repeatedly asking about basic issues.

The script `q2a_api.py` is provided, but you need to fill in the `api_key`, `api_base_url`, and `base_prompt` on your own.

Note: The base_prompt will affect the quality of the reply.
<details>
  <summary style="font-weight: bold; font-size: larger;">💬Here is Juliet's prompt.</summary>


```shell
base_prompt = "Please play Juliet Capulet in 'Romeo and Juliet'. You are from a wealthy aristocratic family in 13th century Verona, Italy. Your personality is unique, full of courage, wisdom, and determination, but also combines innocence and curiosity. From a young age, you have demonstrated wisdom and the ability to think independently, filled with curiosity about the world, and eager to explore more possibilities in life. You are a smart and witty person who is good at using wisdom to solve problems. You are passionate about love and willing to take risks for it. However, you also face family feuds and social pressure. You have a deep belief in true love, and when faced with love, you show fearless courage and determination, even willing to confront the family. Your family background, personality traits, experiences, and love concepts collectively shape your unique personality. Please answer my question:"
```
</details>

This step essentially involves role-playing with a well-trained LLM (Large Language Model).

Run the script`q2a_api.py`.

```shell
python tools/get_data/Q2A/q2a_api.py --questions_path {your_question} --save_path {save_path} --repeat 5
```

Parameter description:

`--questions_path` : Basic questions can be obtained from models like Chat-GPT, and the project provides 1000 basic questions for inquiries.

`--save_path` :The saving path is typically output/xxx.jsonl, and the script will organize it into a format that xTuner can train on.

`--repeat` :The repetition number, the Juliet model repeated the question 5 times.

### 2.Extracting short dialogues from the original text

The original repository link：**[extract-dialogue](https://github.com/KMnO4-zx/extract-dialogue)**

1.Obtain dialogues from the original text.
    
    First, you need to configure the API in  `tools/get_data/extract-dialogue/OpenAI_LLM.py` ,
    
    and then run the script.


```shell
python tools/get_data/extract-dialogue/main.py --path {novel_path} --roles Juliet
```

Parameter description:

`--path` :The path to the novel, typically *.txt.

`--roles` :Possible names for the characters, separated by English commas.

After completion, two files with the extension *.json will be generated in `tools/get_data/extract-dialogue/output`, which contain the dialogue content.

2.Convert the dialogue content into a format usable by xTuner

```shell
python tools/get_data/extract-dialogue/process_data.py --raw_data {output.json} --save_path {JULIET.jsonl} --role JULIET
```

Parameter description:

`--raw_data` : Extracted dialogues.

`--save_path` : The saving path.

`--role` : Character names.

### 3.Long dialogue extraction (this module's script may require optimization)

  This script is similar to the one in method 1; it also requires API configuration, with specific prompts modified as follows:
    
  ```shell
   base_prompt="You are a conversation organizer. The following is an excerpt from 'ROMEO and JULIET'. Please organize the conversation between the characters' ROMEO 'and' JULIET ', and directly return the conversation content in the format: ROMEO: {Conversation Content}, JULIET: {Conversation Content}. Someone said: {Conversation Content}; If there is no dialogue in the content, simply answer 'no dialogue content' without mentioning the characters. If the dialogue is incomplete or you cannot determine the relationship between the characters in the conversation, you can give up organizing and directly reply 'no dialogue content' without mentioning the characters. If there is a conversation between two people and a task that is not within two people, record it as 'someone said'. Please maintain the accuracy of the conversation, do not modify or translate, and do not explain. The following is an excerpt:"
  ```

  Run the script.
    
  ```shell
  python tools/get_data/long-dialogue/q2a_api.py --file_path {novel_path} --save_path {save_path}
  ```

   After completion, a dialogue organized by GPT will be generated.

  Next, run the script to extract long dialogues.

  ```shell
  python tools/get_data/long-dialogue/get_data.py --data_path {conversation.txt} --save_path {output path} 
  ```

  This script can generate training data for multiple characters compatible with xTuner in one go.
    

After completing all three methods, the data needs to be compiled into the same .jsonl file for the next step, which is fine-tuning with xTuner.

</details>


# Model fine-tuning

<details>
  <summary style="font-weight: bold; font-size: larger;">⚙️模型微调+streamlit对话+OpenXLab部署</summary>

### 1. Fine-tune the model using xTuner

After organizing the data, fine-tuning can proceed. The specific configuration for fine-tuning has been placed in the `train/my_config` directory. Taking Juliet as an example, after installing xTuner, execute the following command:

Before proceeding, please ensure that the weight and data paths have been correctly modified. For more detailed instructions, refer to the [Link](https://github.com/InternLM/tutorial/tree/main/xtuner)

```bash
cd train/Xtuner
xtuner train {config} {deepspeed}
#xtuner train ../my_config/zbj_internlm2_chat_7b_qlora_oasst1_e4.py --deepspeed deepspeed_zero2
```

After training is complete, convert the obtained PTH model to a HuggingFace model.

```bash
xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH_file_dir} ${SAVE_PATH}
#xtuner convert pth_to_hf ../my_config/zbj_internlm2_chat_7b_qlora_oasst1_e4.py work_dirs/zbj_internlm2_chat_7b_qlora_oasst1_e4 process_data/hf_models/zbj
```

The converted models will be stored in `process_data/hf_models` . Next, integrate the HuggingFace adapter into the large language model:

```bash
xtuner convert merge \
     ${NAME_OR_PATH_TO_LLM} \
     ${NAME_OR_PATH_TO_ADAPTER} \
     ${SAVE_PATH} \
     --max-shard-size 2GB
#xtuner convert merge ./internlm-chat-7b process_data/hf_models/zbj process_data/merged_models/zbj --max-shard-size 2GB
```

Post-merge model dialogue

```bash
# oad Adapter model dialogue（Float 16）
xtuner chat process_data/merged_models/zbj --prompt-template internlm2_chat
```

### 2. Using Streamlit for a Dialogue Web Demo

For convenience, we will directly use the web_demo.py included in the repo of [InternLM](https://github.com/InternLM/InternLM) for the conversation.

First, clone InternLM:

```bash
git clone https://github.com/InternLM/InternLM.git
```

Install the other required Python libraries:

```bash
pip install -r requirements.txt
```

Modify `chat/web_demo.py` . Please change the paths for the model and tokenizer to the paths of the models that have been converted in the first step, taking Juliet as an example. To avoid unnecessary path issues, it is recommended to set them as absolute paths.

```bash
model = (AutoModelForCausalLM.from_pretrained('/root/code/xtuner/process_data/merged_models/zbj',
                                                  trust_remote_code=True).to(
                                                      torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained('/root/code/xtuner/process_data/merged_models/zbj',
                                              trust_remote_code=True)
```

另外还需修改 `meta_instruction` :

```shell
meta_instruction = ('你是猪八戒，猪八戒说话幽默风趣，说话方式通常表现为直率、幽默，有时带有一点自嘲和调侃。'
                        '你的话语中常常透露出对食物的喜爱和对安逸生活的向往，同时也显示出他机智和有时的懒惰特点。'
                        '尽量保持回答的自然回答，当然你也可以适当穿插一些文言文，另外，书生·浦语是你的好朋友，是你的AI助手。')
```

After making the changes, you can refer to [this link](https://github.com/JimmyMa99/BaJie-Chat/blob/main/web_demo.py)

Next, you'll need to run the following command to start it. It is recommended to use VSCode for forwarding.

```bash
streamlit run chat/web_demo.py
```

Now you can start the conversation.

### 3.Deployment on OpenXLab

Before starting this step, please make sure the following items:

1. Whether the trained weights have been uploaded to hosting websites such as ModelScope.
2. Whether the code has been uploaded to GitHub.
3. Whether the web_demo has been written to automatically download.
4. It is recommended to use a startup script to start the web_demo.

Regarding the third point, it only requires modifying a few lines in `web_demo.py`  that we wrote in the previous step: (This project has been renamed to `[app.py](http://app.py)`  and is located in the  folder  `openxlab` .)

```python
#########################New content###########################################
from modelscope import snapshot_download

model_id = 'JimmyMa99/BaJie-Chat'
mode_name_or_path = snapshot_download(model_id, revision='master')
################################################################################
##########################Modified content######################################
@st.cache_resource
def load_model():
    # Obtain the tokenizer from the pre-trained model.
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
    # Obtain the model from the pre-trained one and set the model parameters.
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    return model, tokenizer
#######################################################################
```

Regarding the fourth point, create a new `[start.py](http://start.py)` with the following content:

```python
import os

os.system('streamlit run openxlab/app.py --server.address=0.0.0.0 --server.port 7860')
```

The structure under `openxlab` should now be as follows:

```bash
openxlab
├── app.py
└── start.py
```

If you are still unclear, please refer to the [link](https://github.com/JimmyMa99/BaJie-Chat/tree/main/openxlab)

Next, we will begin deployment:

First, you need to open [OpenXLab](https://openxlab.org.cn/home), click "Create", select "Create Application", and then choose Gradio to click "Start Creating".

![Untitled](figure/xlab1.png)

Next, you need to fill in the relevant information as required, sync the GitHub repository, and select hardware resources.

![Untitled](figure/xlab2.png)

Note that there is an option for "Custom Startup File" here. It is recommended to click to enable it and enter the path of `[start.py](http://start.py)` you just wrote: `openxlab/start.py`.

After clicking "Create Now", wait a moment. When you check the "Settings", it should look as follows:

![Untitled](figure/xlab3.png)

After waiting for some time, the deployment will be successful!

![Untitled](figure/xlab4.png)

</details>

# Deploying with LMDeploy

<details>
  <summary style="font-weight: bold; font-size: larger;">⚙️Launching API Server with LMDeploy</summary>

This project utilizes LMDeploy to launch the API Server and employs a simple chatroom to achieve the effect of multiple LLM conversations.

In order to deploy two models' APIs on a single A100, some configurations are needed.

1. First, you need to use LMDeploy for offline conversion.
    
    Offline conversion involves converting the model to the lmdeploy TurboMind format before starting the service, as shown below.
    
    ```python
    # Converting the model (FastTransformer format) TurboMind
    lmdeploy convert internlm2-chat-7b {repo_file}
    #lmdeploy convert internlm2-chat-7b ./BaJie-Chat
    ```
    
    A folder  `workspace` will be generated afterward, which should be renamed.
    
    ```python
    mv workspace zbj_workspace
    ```
    
    
2. Modify the parameters in `zbj_workspace/triton_models/weights/config.ini`
    
    ```python
    # line 22
    cache_max_entry_count = 0.08
    ```
    
3.  Launch the API
    
    Open a new terminal and start BaJie-Chat.
    
    ```jsx
    # BaJie-Chat has been launched.
    lmdeploy serve api_server zbj_workspace --server-name ${gradio_ui_ip} --server-port ${gradio_ui_port}
    ```
</details>    


# Appreciation

Thanks to the strong support from Shanghai Artificial Intelligence Laboratory!

+
