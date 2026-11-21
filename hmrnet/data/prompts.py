import json
import random

from ..models.text_encoder import PromptBank


class PromptSampler:
    """Builds the prompt set used by the CEM for each training image.

    Prompts of the classes present in the image come first (class-name prompt and,
    when available, descriptive phrases), followed by prompts of other seen classes
    until ``max_prompts`` is reached.
    """

    def __init__(self, label_space, bank, max_prompts=16, phrases_file=None, seen_labels=None, template=None):
        self.label_space = label_space
        self.bank = bank
        self.max_prompts = max_prompts
        self.template = template or "an aerial image of a {}."
        self.phrases = {}
        if phrases_file:
            with open(phrases_file, "r", encoding="utf-8") as f:
                self.phrases = json.load(f)
        self.seen = seen_labels or list(range(1, len(label_space) + 1))

    def class_prompt(self, label):
        return PromptBank.class_prompt(self.label_space.names[label - 1], self.template)

    def prompts_for(self, labels):
        present = list(dict.fromkeys(int(l) for l in labels))
        prompts = []
        for l in present:
            prompts.append(self.class_prompt(l))
            name = self.label_space.names[l - 1]
            if name in self.phrases and self.phrases[name]:
                prompts.append(random.choice(self.phrases[name]))
        negatives = [l for l in self.seen if l not in present]
        random.shuffle(negatives)
        for l in negatives:
            if len(prompts) >= self.max_prompts:
                break
            prompts.append(self.class_prompt(l))
        return prompts[: self.max_prompts]

    def batch_embeddings(self, targets, device):
        return [self.bank.get(self.prompts_for(t["labels"].tolist())).to(device) for t in targets]

    def class_embeddings(self, labels):
        return self.bank.get([self.class_prompt(l) for l in labels])
