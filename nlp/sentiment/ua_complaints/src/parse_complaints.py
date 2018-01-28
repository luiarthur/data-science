import os
from BeautifulSoup import BeautifulSoup

def read_file(path):
    f = open(path, 'r')
    contents = f.read()
    f.close()
    return contents

def parse_complaint(html):
    soup = BeautifulSoup(''.join(html))
    complaints = []
    ratings = []
    for hit in soup.findAll(attrs={'class' : 'review-body__text'}):
        complaints.append(hit.contents[0].text.encode('utf-8'))

    for hit in soup.findAll(attrs={'itemprop' : 'ratingValue'}):
        ratings.append(hit)

    ratings = map(lambda x: int(x['content']), ratings[1:])
    assert(len(complaints) == len(ratings))

    return zip(complaints, ratings)

### Test ###
html = read_file("../dat/html/1.html")
ua_complaints = parse_complaint(html)

### TODO: do this for all files, 1.html, 2.html, ..., 70.html
### TODO: bi-grams, tri-grams. Logistic regression. ratings ~ complaints

# MAIN
#parse_complaints('../dat/html/', '../dat/complaints/')
