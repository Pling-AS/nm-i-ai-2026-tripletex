with open("task-1-norgesgruppen/submission/infer_cls.py", "r") as f:
    content = f.read()

content = content.replace(".eval()", ".train(False)")

with open("task-1-norgesgruppen/submission/infer_cls.py", "w") as f:
    f.write(content)
