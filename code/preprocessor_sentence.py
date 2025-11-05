# coding=utf-8

import re
import os
from bs4 import BeautifulSoup

# 输入输出路径
input_folder = "source"
output_folder = 'DataForDL'

# 配置参数
MAX_SENTENCE_LENGTH = 400  # 最大句子长度（按字符数）

def split_chinese_sentences(text, filename):
    """精准分句函数（保留中英文空格）"""
    # Step 1: 保护中英文间空格和医学缩写
    protected = re.sub(r'([a-zA-Z]) ( [a-zA-Z])', r'\1__EN_SPACE__\2', text)
    medical_abbreviations = [r'COVID-19', r'IgG', r'pH']
    for abbr in medical_abbreviations:
        protected = re.sub(abbr, lambda m: m.group().replace('-', '__HYPHEN__'), protected)

    # Step 2: 处理空白和换行符
    processed = re.sub(r'[\r\t]+', ' ', protected)          # 替换 \r 和 \t 为空格
    processed = re.sub(r'[ ]{2,}', ' ', processed)          # 合并多个空格
    processed = re.sub(r'\n+', '\n', processed)             # 合并多个换行符为单个

    


    # Step 4: 分句（使用正则 + 换行符）
    sentence_delimiters = re.compile(
        r'([。！？]|\. |\n)(?![\]）】\)\"])'
    )
    sentences = []
    last_pos = 0
    for match in sentence_delimiters.finditer(processed):
        end_pos = match.end()
        sentence = processed[last_pos:end_pos].strip()
        # 过滤空句子（如仅包含换行符）
        if sentence:
            sentences.append(sentence)
        last_pos = end_pos
    if last_pos < len(processed):
        sentences.append(processed[last_pos:].strip())

    # Step 5: 恢复保护的内容
    restored = [s.replace('__EN_SPACE__', ' ') for s in sentences]
    restored = [s.replace('__HYPHEN__', '-') for s in restored]
    
    # Step 6: 分割超长句子
    final = []
    for s in restored:
        while len(s) > MAX_SENTENCE_LENGTH:
            split_pos = max(
                s.rfind('，', 0, MAX_SENTENCE_LENGTH),
                s.rfind('；', 0, MAX_SENTENCE_LENGTH),
                s.rfind('。', 0, MAX_SENTENCE_LENGTH)
            )
            if split_pos == -1:
                split_pos = MAX_SENTENCE_LENGTH
            final.append(s[:split_pos+1].strip())
            print(f"长句分割 [{filename}] (长度={len(s)}): {s[:50]}")
            s = s[split_pos+1:]
        final.append(s)
    return final

def preprocess_markdown_html_text(text, filename):
    """精准预处理函数"""
    # 去除HTML标签
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text()
    
    # 精确替换非图片链接为(链接)
    cleaned = re.sub(
        r'\[([^\]]+)\]\(((?!http[s]?://[^\s]+?\.(jpg|png|jpeg|gif))[^\)]+)\)',
        r'\1(链接)',
        cleaned
    )
    
    # 完全删除图片链接（包括base64格式）
    cleaned = re.sub(r'!\[.*?\]\([^\)]*\)', '', cleaned)
    cleaned = re.sub(r'!\[\]\((data:image/[^;]+;base64,[^\)]+)\)', '', cleaned)
    
    # 清理其他Markdown标记
    cleaned = re.sub(r'#+\s?', '', cleaned)
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    
    return split_chinese_sentences(cleaned, filename)

if __name__ == '__main__':
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            with open(input_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            processed = preprocess_markdown_html_text(raw_text, filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(processed))


