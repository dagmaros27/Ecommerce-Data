import re
from typing import List, Tuple
import pandas as pd
import string

def load_raw_telegram_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def clean_amharic_text(text: str) -> str:
    if pd.isnull(text):
        return ""

    text = text.replace("፦", " ")
    text = re.sub(r'(?<!\s)(09\d{8})(?!\s)', r' \1 ', text)
    
    text = re.sub(fr'([\u1200-\u137F])(\d)', r'\1 \2', text)
    text = re.sub(fr'(\d)([\u1200-\u137F])', r'\1 \2', text)
    text = re.sub(r'[^\w\s፡።፣፤፥፦መሀቀለዐነአሰቀቸበኸተሰከወፈ]+', '', text)

    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def amharic_tokenize(text: str) -> list:
    return text.split()

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    df = df[df["Message"].notnull() & df["Message"].str.strip().ne("")]

    df["cleaned_message"] = df["Message"].apply(clean_amharic_text)

    df["tokens"] = df["cleaned_message"].apply(amharic_tokenize)

    return df[["ID", "Channel Title", "Channel Username", "Date", "cleaned_message", "tokens"]]

def save_cleaned_data(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)
    
    
    

# Regular expressions for matching contact, price, and location patterns
# Adjusted to match Amharic text and common patterns
price_pattern = re.compile(r'(?:ዋጋ[:：]?\s*)?(\d{2,6})(?:\s*ብር|birr)?', re.IGNORECASE)
contact_pattern = re.compile(r'09\d{8}|\+251\d{9}')
location_keywords = ['ቦታ', 'አድራሻ', 'ሞል', 'ቢሮ', 'ህንፃ', 'ሱቅ', 'ፎቅ', 'ፊት', 'ቤተክርስቲያን', 'መዳህኒዓለም']
price_keywords = ['ዋጋ', 'ብር', 'ቅናሽ', 'ብር ']
product_keywords = [
    'juicer', 'set', 'dispenser', 'ጁስ', 'ማሽን', 'portable', 'ብርጭቆ',
    'የፀጉር', 'GROOMING', 'የማፅዳት', 'Ergonomic', 'Multi', '3in1',
    'Mop', 'Slicer', 'Gloves', 'Humidifier', 'Phone', 'Smart'
]

def label_tokens_bio(tokens: List[str]) -> List[Tuple[str, str]]:
    """Label tokens with BIO scheme for NER."""
    labels = []
    prev_label = 'O'
    prev_type = None

    for token in tokens:
        label = 'O'
        entity_type = None

        # CONTACT
        if re.match(contact_pattern, token):
            entity_type = "CONTACT"
        # PRICE
        elif any(kw in token for kw in price_keywords) or re.match(price_pattern, token):
            entity_type = "PRICE"
        # LOCATION
        elif any(loc_kw in token for loc_kw in location_keywords):
            entity_type = "LOCATION"
        # PRODUCT
        elif any(prod_kw.lower() in token.lower() for prod_kw in product_keywords):
            entity_type = "PRODUCT"

        if entity_type:
            # Use B- if previous label was different entity or O, else I-
            if prev_label == 'O' or prev_type != entity_type:
                label = f"B-{entity_type}"
            else:
                label = f"I-{entity_type}"
        else:
            label = "O"

        labels.append((token, label))
        prev_label = label
        prev_type = entity_type if entity_type else None

    return labels

def dataframe_to_conll(df: pd.DataFrame, token_column: str = 'tokens') -> List[str]:
    """Convert DataFrame tokens to CoNLL format."""
    conll_lines = []
    for _, row in df.iterrows():
        try:
            tokens = row[token_column]
            if isinstance(tokens, str):
                tokens = eval(tokens)
            labeled = label_tokens_bio(tokens)
            for token, tag in labeled:
                conll_lines.append(f"{token} {tag}")
            conll_lines.append("")  # Sentence separator
        except Exception as e:
            print(f"Error processing row: {e}")
    return conll_lines

def save_conll_file(conll_lines: List[str], output_path: str):
    """Save CoNLL-formatted lines to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in conll_lines:
            f.write(line + '\n')