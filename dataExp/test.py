from sets import Set


def answer(document, searchTerms):
    print type(document)
    print type(searchTerms)
    searchset = Set(searchTerms)
    print searchset
    doclist = document.split()
    length = []
    for i in range(len(doclist)):
        tmpset = searchset.copy()
        if doclist[i] in tmpset:
            tmpset.remove(doclist[i])
            j = i+1
            while j < len(doclist) and len(tmpset) > 0:
                if doclist[j] in tmpset:
                    tmpset.remove(doclist[j])
                j += 1
            if len(tmpset) == 0:
                length.append(j-i)
            else:
                length.append(-1)
        else:
            length.append(-1)

    min = 501
    for i in range(len(length)):
        if length[i]>0 and length[i]<min:
            min = length[i]
            minpos = i
    res=""
    for i in range(min):
        res += doclist[minpos+i]
        res += " "
    ans = str(res[0:len(res)])
    print type(ans)
    return ans


document = "a b c d a"
searchTerms = ["a","c","d"]
document = "many google employees can program"
searchTerms = ["google", "program"]
document = "world there hello hello where world"
searchTerms = ["hello","world"]

#document = "a"
#searchTerms = ["a"]

print answer(document,searchTerms)