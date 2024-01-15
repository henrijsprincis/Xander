output=""
all_prompts = []

def remove_empty(lines):
    good_lines = []
    for line in lines:
        if line != "\n":
            good_lines.append(line)
    return good_lines
for i in range(200):
    with open('bigboi'+str(i)+'.txt','r') as file:#bigboiNoExmp
        #split by (;), then start at first SELECT.
        lines = file.readlines()
        lines = remove_empty(lines)
        lines = "".join(lines)#joined
        lines = lines.split(";")#split by ;
        if len(lines)!=6:
            print("Error: ",i)
            all_prompts.append(i)
        lines = lines[:-1]
        for idx, line in enumerate(lines):
            line = line.lower()
            start_idx = line.find("select")
            if start_idx == -1:
                print("Error: ", i)
                all_prompts.append(i)
            line = line.replace("\n", " ")
            lines[idx] = line[start_idx:]+";"#add in the semicolon

        output+="\n".join(lines)+"\n"
print(all_prompts)

with open('allQueriesExamples.txt','w') as file:
    lines = file.write(output)