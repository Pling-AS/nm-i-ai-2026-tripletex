with open("task-1-norgesgruppen/submission/run.py", "r") as f:
    content = f.read()

content = content.replace("""        if len(fused["scores"]) == 0:
            continue

        if len(fused["scores"]) == 0:
            continue""", """        if len(fused["scores"]) == 0:
            continue""")

with open("task-1-norgesgruppen/submission/run.py", "w") as f:
    f.write(content)
