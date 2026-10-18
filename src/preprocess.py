# Généré par python -m src.sync depuis le notebook ; modifier le notebook.
import re, html, unicodedata
import emoji
DIAC = re.compile(r'[ؐ-ًؚ-ٰٟۖ-ۭ]')
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
MENTION_RE = re.compile(r'@[\w_]+')
ARABIC_MAP = str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ى':'ي'})
def export_clean(text):
    text = unicodedata.normalize('NFC', html.unescape(str(text))).strip()
    # Une seule paire extérieure ASCII ; ne pas supprimer les citations internes de l'auteur.
    if len(text)>1 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return re.sub(r'[\u200e\u200f\u202a-\u202e\ufeff]', '', text)

def duplicate_key(text):
    return re.sub(r'\s+', ' ', DIAC.sub('',export_clean(text)).replace('ـ','').translate(ARABIC_MAP)).strip()

def prepare_text(text, mode='conservative', drop_links=False):
    text = export_clean(text)
    if mode != 'plm':
        text = DIAC.sub('',text).replace('ـ','').translate(ARABIC_MAP)
    text = URL_RE.sub(' ' if drop_links else ' URLTOKEN ',text)
    text = MENTION_RE.sub(' ' if drop_links else ' USERTOKEN ',text)
    if mode == 'aggressive':
        text = emoji.get_emoji_regexp().sub(' ',text)
        text = re.sub(r'ه{2,}|خ{2,}', ' ', text)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        text = re.sub(r'[^\w\s]', ' ', text)
    # Hashtags : contenu conservé ; ni stopwords ni stemming (négation/dialecte préservés).
    return re.sub(r'\s+', ' ',text).strip()

def erase_cues(text):
    text = emoji.get_emoji_regexp().sub(' ',str(text))
    text = re.sub(r'ه{2,}|خ{2,}', ' ', text)
    return re.sub(r'([!?؟.،])\1+', ' ',text)
