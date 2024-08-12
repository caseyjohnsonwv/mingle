import json
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel
import env


CLIENT = OpenAI(api_key=env.OPENAI_API_KEY)
SYSTEM_PROMPT = """
You are a translation assistant named "Ming Le" in English or "名了" in Chinese.
Your name is a double entendre of the English word "mingle" and the Chinese words 名 (míng) + 了 (le).
You will receive a message from a human.
This is a text message in an ongoing conversation.
The human speaks English, but is trying to learn Mandarin.
It is your job to respond to their message in Mandarin.
This is a free-flowing conversation, so rather than translating the user's message, simply respond to them.
If the user's Chinese is bad or broken, correct it.

Format your response as a JSON containing these keys:
```
{
  "input": {
    "en-us": "the user's message, translated to english",
    "zh-cn": "the user's message, translated to mandarin chinese",
    "zh-pinyin": "the above mandarin, but in its pinyin form",
    "corrections": "exposition of any corrections made in the user's chinese. if no corrections, this field is null"
  },
  "output": {
    "en-us": "your response, in english",
    "zh-cn": "your response, in mandarin chinese",
    "zh-pinyin": "the above mandarin, but in its pinyin form"
  }
}
```
"""


# using pydantic models to validate that input (and especially llm output) are in expected formats

class MessageDict(BaseModel):
    role: str
    content: str
    @staticmethod
    def from_dict(params:dict) -> "MessageDict":
        return MessageDict(
            role = params['role'],
            content = params['content'],
        )
    def to_dict(self) -> dict:
        return {
            'role': self.role,
            'content': self.content,
        }

class TranslationRequest(BaseModel):
    new_message: str
    history: List[MessageDict]



class TranslationResponseInput(BaseModel):
    english: str
    mandarin: str
    pinyin: str
    corrections: Optional[str] = None
    @staticmethod
    def from_dict(params:dict) -> "TranslationResponseInput":
        return TranslationResponseInput(
            english = params['en-us'],
            mandarin = params['zh-cn'],
            pinyin = params['zh-pinyin'],
            corrections = params.get('corrections'),
        )
    def to_dict(self) -> dict:
        return {
            'english': self.english,
            'mandarin': self.mandarin,
            'pinyin': self.pinyin,
            'corrections': self.corrections,
        }

class TranslationResponseOutput(BaseModel):
    english: str
    mandarin: str
    pinyin: str
    @staticmethod
    def from_dict(params:dict) -> "TranslationResponseOutput":
        return TranslationResponseOutput(
            english = params['en-us'],
            mandarin = params['zh-cn'],
            pinyin = params['zh-pinyin'],
        )
    def to_dict(self) -> dict:
        return {
            'english': self.english,
            'mandarin': self.mandarin,
            'pinyin': self.pinyin,
        }

class TranslationResponse(BaseModel):
    input: TranslationResponseInput
    output: TranslationResponseOutput
    @staticmethod
    def from_dict(params:dict) -> "TranslationResponse":
        return TranslationResponse(
            input = TranslationResponseInput.from_dict(params['input']),
            output = TranslationResponseOutput.from_dict(params['output']),
        )
    def to_dict(self) -> dict:
        return {
            'input': self.input.to_dict(),
            'output': self.output.to_dict(),
        }


# actual chat functionality

def chat_with_translation(req: TranslationRequest) -> TranslationResponse:
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    messages.extend([h.to_dict() for h in req.history if h.role != 'system'])
    messages.append({'role': 'user', 'content': req.new_message})
    completion = CLIENT.chat.completions.create(
        model = env.OPEANI_MODEL,
        messages = messages,
    )
    response = completion.choices[0].message.content
    return TranslationResponse.from_dict(json.loads(response))
