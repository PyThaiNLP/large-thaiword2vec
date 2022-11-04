from gensim.models import Word2Vec
import wandb
from gensim.models.callbacks import CallbackAny2Vec
import os
import pickle
import re
import multiprocessing
from tqdm.auto import tqdm
from pythainlp import thai_letters, thai_punctuations, thai_symbols
thai_characters = "".join(
    [thai_letters, thai_punctuations, thai_symbols]
)
_n=list(thai_letters+thai_punctuations+thai_symbols)
_n.remove("ๆ")
thai_characters_witout_d = "".join(_n)

wandb.init(project="thai-w2v-new",
           config={
           })

from pythainlp.tokenize import word_tokenize
from thai2fit_preprocess import (
    fix_html,
    lowercase_all,
    remove_space,
    replace_rep_after,
    replace_rep_nonum,
    replace_url,
    replace_wrep_post,
    replace_wrep_post_nonum,
    rm_brackets,
    rm_useless_newlines,
    rm_useless_spaces,
    spec_add_spaces,
    ungroup_emoji,
)
from pythainlp.util import reorder_vowels
pre_rules_th = [
    replace_rep_after,
    fix_html,
    reorder_vowels,
    spec_add_spaces,
    rm_useless_spaces,
    rm_useless_newlines,
    rm_brackets,
    replace_url,
]
def fix_some_rule(text):
    text = text.replace('\u200b','')
    text = text.replace('\u2022', ' ')
    text = text.replace('\xad', '')
    text = text.replace('\u00ad', '')
    text = text.replace('\N{SOFT HYPHEN}', '')
    text =re.sub("([{t}])(ๆ)".format(t=thai_characters_witout_d),"\\1 \\2",text)
    text =re.sub("(ๆ)([{t}])".format(t=thai_characters_witout_d),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\,)".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\,)([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub('(\")([\w{t}])'.format(t=thai_characters),"\\1 \\2",text)
    text =re.sub('([\w{t}])(\")'.format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w])(\,)","\\1 \\2",text)
    text =re.sub("(\,)([\w])","\\1 \\2",text)
    text = re.sub("(\()([\w\dก-๙])","\\1 \\2",text)
    text =re.sub("([\w\dก-๙])(\))","\\1 \\2",text)
    text = re.sub("([\w\dก-๙])(\()","\\1 \\2",text)
    text =re.sub("(\))([\w\dก-๙])","\\1 \\2",text)
    text = re.sub("([\w\dก-๙])(\.{2,})","\\1 \\2",text)
    text = re.sub("(\.{2,})([\w\dก-๙])","\\1 \\2",text)
    text =re.sub("([\w{t}])(\[)".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\[)([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\])([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\“)([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\“)".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\”)".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\”)([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\')".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\')([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\’)".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\')".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\’)([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\')([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([{t}])(\:)".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\:)([{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\;)([{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\;)".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\?)".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\?)([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\!)".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\!)([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\‘)".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\‘)([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\|)".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\|)([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\w{t}])(\%)".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\%)([\w{t}])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("(\.)([\]\[\)\(])".format(t=thai_characters),"\\1 \\2",text)
    text =re.sub("([\(\)\]\[])(\.)".format(t=thai_characters),"\\1 \\2",text)
    text=text.replace(")⇒",") ⇒")
    text = re.sub("([\w\dก-๙])(\.{2,})","\\1 \\2",text)
    text = re.sub("(\.{2,})([\w\dก-๙])","\\1 \\2",text)
    text = re.sub("""([`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,/<>?])([\dก-๙])""","\\1 \\2",text)
    text = re.sub("""([\dก-๙])([`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,/<>?])""","\\1 \\2",text)
    #text = re.sub("""(?<=[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?])([`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?])\1*""", " \g<0>",text)
    return text
post_rules_th = [replace_wrep_post, ungroup_emoji, lowercase_all]

# sparse features
pre_rules_th_sparse = [fix_some_rule]+pre_rules_th[1:] + [replace_rep_nonum]
post_rules_th_sparse = post_rules_th[1:] + [
    replace_wrep_post_nonum,
    remove_space,
]

def thai_word_tokenize(text):
    return word_tokenize(text,engine="newmm")

def process_thai(
    text: str,
    pre_rules = pre_rules_th_sparse,
    tok_func = thai_word_tokenize,
    post_rules = post_rules_th_sparse,
):
    """
    Process Thai texts for models (with sparse features as default)
    :param str text: text to be cleaned
    :param list[func] pre_rules: rules to apply before tokenization.
    :param func tok_func: tokenization function (by default, **tok_func** is
                          :func:`pythainlp.tokenize.word_tokenize`)
    :param list[func]  post_rules: rules to apply after tokenizations
    :return: a list of cleaned tokenized texts
    :rtype: list[str]
    :Note:
      - The default **pre-rules** consists of :func:`fix_html`,
        :func:`pythainlp.util.normalize`,
        :func:`spec_add_spaces`,
        :func:`rm_useless_spaces`,
        :func:`rm_useless_newlines`,
        :func:`rm_brackets`
        and :func:`replace_rep_nonum`.
      - The default **post-rules** consists of :func:`ungroup_emoji`,
        :func:`lowercase_all`,  :func:`replace_wrep_post_nonum`,
        and :func:`remove_space`.
    :Example:
        1. Use default pre-rules and post-rules:
        >>> from pythainlp.ulmfit import process_thai
        >>> text = "บ้านนนนน () อยู่นานนานนาน 😂🤣😃😄😅 PyThaiNLP amp;     "
        >>> process_thai(text)
        [บ้าน', 'xxrep', '   ', 'อยู่', 'xxwrep', 'นาน', '😂', '🤣',
        '😃', '😄', '😅', 'pythainlp', '&']
        2. Modify pre_rules and post_rules arugments with
           rules provided in :mod:`pythainlp.ulmfit`:
        >>> from pythainlp.ulmfit import (
            process_thai,
            replace_rep_after,
            fix_html,
            ungroup_emoji,
            replace_wrep_post,
            remove_space)
        >>>
        >>> text = "บ้านนนนน () อยู่นานนานนาน 😂🤣😃😄😅 PyThaiNLP amp;     "
        >>> process_thai(text,
                         pre_rules=[replace_rep_after, fix_html],
                         post_rules=[ungroup_emoji,
                                     replace_wrep_post,
                                     remove_space]
                        )
        ['บ้าน', 'xxrep', '5', '()', 'อยู่', 'xxwrep', '2', 'นาน', '😂', '🤣',
         '😃', '😄', '😅', 'PyThaiNLP', '&']
    """
    res = text

    for rule in pre_rules:
        res = rule(res)
    res = tok_func(res)
    for rule in post_rules:
        res = rule(res)

    return res


with open('big-raw.pickle', 'rb') as handle:
    new_train = pickle.load(handle)
data_all=[]
with open("save-cuted-ok.txt","w",encoding="utf-8") as f:
    for i in tqdm(list(range(len(new_train)))): #len(new_train)
        _temp = process_thai(new_train[i])
        data_all.append(_temp)
        f.write(str(_temp)+"\n")
del new_train


# init callback class
class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0
        self.best_loss = 1000000000
        self.loss_previous_step = 100000000000

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            wandb.log({'epoch':self.epoch,'loss':loss,'loss_fix':loss})
        else:
            wandb.log({'epoch':self.epoch,'loss':loss,'loss_fix':loss- self.loss_previous_step})
        model.save(os.path.join("ok",'5-best-epoch-{}.bin'.format(self.epoch)))
        self.epoch += 1
        self.loss_previous_step = loss

model = Word2Vec(data_all, vector_size=400, window=5, min_count=5, workers=multiprocessing.cpu_count()-1,compute_loss=True,epochs=50,callbacks=[callback()])
wandb.finish()