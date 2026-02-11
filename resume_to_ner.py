def convert_resume_to_ner(text, skill_library):

    samples = []
    sentences = text.split("。")

    for sentence in sentences:
        if len(sentence.strip()) < 5:
            continue

        tokens = list(sentence)
        labels = ["O"] * len(tokens)

        for skill in skill_library:
            if skill in sentence:
                start = sentence.index(skill)
                for i in range(len(skill)):
                    labels[start+i] = "B-SKILL" if i == 0 else "I-SKILL"

        samples.append({
            "tokens": tokens,
            "labels": labels
        })

    return samples
