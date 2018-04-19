import ply.lex as lex
from nltk import ngrams
import collections
# List of token names.   This is always required
tokens = (
   'LINE_COMMENT',
   'AREA_COMMENT',
   'AREA_COMMENT_CONTINUE',
   'NUMBER',
   'OPERATOR',
   'QUOTE',
   'CHAR',
   'CHAR_CONTINUE',
   'DIRECTIVE',
   'IDENTIFIER',
   'OPEN_PAREN',
   'CLOSE_PAREN',
   'OPEN_CURLY',
   'CLOSE_CURLY',
   'OPEN_SQUARE',
   'CLOSE_SQUARE'
)

def t_LINE_COMMENT(t):
    r'\/\/[^\n]*'
    return t

def t_AREA_COMMENT(t):
    r'\/\*([^*]|\*(?!\/))*\*\/'
    return t

def t_AREA_COMMENT_CONTINUE(t):
    r'\/\*([^*]|\*(?!\/))*\*'
    return t

# A regular expression rule with some action code
def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_OPERATOR(t):
    #r'[-\*/\+><=]'
    r'([-<>~!%^&*\/+=?|.,:;]|->|<<|>>|\*\*|\|\||&&|--|\+\+|[-+*|&%\/=]=)'
    return t

def t_QUOTE(t):
    r'"([^"\n]|\\")*"?'
    return t

def t_CHAR(t):
    r'\'(\\?[^\'\n]|\\\')\''
    return t

def t_CHAR_CONTINUE(t):
    r'\'[^\']*'
    return t

def t_DIRECTIVE(t):
    r'\#(\S*)'
    return t

def t_IDENTIFIER(t):
    r'[_A-Za-z][0-9_A-Za-z]*'
    return t

t_OPEN_PAREN = r'\('
t_CLOSE_PAREN = r'\)'
t_OPEN_CURLY = r'{'
t_CLOSE_CURLY = r'}'
t_OPEN_SQUARE = r'\['
t_CLOSE_SQUARE = r'\]'

# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# A string containing ignored characters (spaces and tabs)
t_ignore  = ' \t'

# Error handling rule
def t_error(t):
    #print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

# Build the lexer
#lexer = lex.lex()

# Test it out
#data = open('test.cpp', 'r').read()

# Give the lexer some input
#lexer.input(data)

cpp_keywords = {
    "alignas",
    "alignof",
    "and",
    "and_eq",
    "asm",
    "atomic_cancel",
    "atomic_commit",
    "atomic_noexcept",
    "auto",
    "bitand",
    "bitor",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "char16_t",
    "char32_t",
    "class",
    "compl",
    "concept",
    "const",
    "constexpr",
    "const_cast",
    "continue",
    "co_await",
    "co_return",
    "co_yield",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "dynamic_cast",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "import",
    "inline",
    "int",
    "long",
    "module",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "not",
    "not_eq",
    "nullptr",
    "operator",
    "or",
    "or_eq",
    "private",
    "protected",
    "public",
    "register",
    "reinterpret_cast",
    "requires",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "static_assert",
    "static_cast",
    "struct",
    "switch",
    "synchronized",
    "template",
    "this",
    "thread_local",
    "throw",
    "true",
    "try",
    "typedef",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "wchar_t",
    "while",
    "xor",
    "xor_eq",
    "{",
    "}"
}

lexer = lex.lex()
def get_freqs_from_cpp_source(source):
    lexer.input(source)
    freqs = {}
    for keyword in cpp_keywords:
        freqs[keyword] = 0

    while True:
        tok = lexer.token()
        if not tok : break
        if valid_token(tok):
            freqs[tok.value] += 1

    return freqs

def valid_token(tok):
    return (tok.type == 'IDENTIFIER' and tok.value in cpp_keywords) or tok.type == 'OPEN_CURLY' or tok.type == 'CLOSE_CURLY'

def get_keywords(source):
    lexer.input(source)
    ans = []
    while True:
        tok = lexer.token()
        if not tok : break
        if valid_token(tok):
            ans.append(tok.value)
    return ans

#print(cpp_keywords)
cpp_keyword_ngrams = {}

def make_ngram_dict(n, ngram):
    if n == 0:
        #print(ngram)
        #print(len(cpp_keyword_ngrams))
        cpp_keyword_ngrams[tuple(ngram)] = 0
        return

    for j in cpp_keywords:
        cp = ngram.copy() #todo: check append - pop
        cp.append(j)
        make_ngram_dict(n - 1, cp)

#print(cpp_keyword_ngrams)
def get_ngram(n, source):
    if len(cpp_keyword_ngrams) == 0:
        make_ngram_dict(n, [])
        print(len(cpp_keyword_ngrams))
    else:
        for key in cpp_keyword_ngrams.keys():
            cpp_keyword_ngrams[key] = 0
    #print(cpp_keyword_ngrams)
    lexer.input(source)

    ngram_buffer = []
    i = 0
    while i < n:
        tok = lexer.token()
        if not tok: break
        if valid_token(tok):
            ngram_buffer.append(tok.value)
            i += 1

    if len(ngram_buffer) != n:
        print(source) 
        print(ngram_buffer)
        raise "Error" #todo: add errors

    cpp_keyword_ngrams[tuple(ngram_buffer)] += 1
    while True:
        tok = lexer.token()
        if not tok: break
        if valid_token(tok):
            cpp_keyword_ngrams[tuple(ngram_buffer)] += 1
            del ngram_buffer[0]
            ngram_buffer.append(tok.value)            

    #print(cpp_keyword_ngrams)
    #ngram = ngrams(source, n)
    #print(ngram)
    #for gram in ngram:
    #    print(gram)
    #    valid_ngram = True
        #for i in gram:
        #    print(i)
         #   if i not in cpp_keywords:
         #       valid_ngram = False
         #       break
    #    if valid_ngram:
    #        cpp_keyword_ngrams[gram] += 1

    #open('log.log', 'w').write(str(cpp_keyword_ngrams))
    return cpp_keyword_ngrams #todo: change

#get_ngram(2, "test.cpp")


#freqs = {}
#for keyword in cpp_keywords:
#    freqs[keyword] = 0

# Tokenize
#while True:
#    tok = lexer.token()
#    if not tok: break      # No more input
#    if tok.type == 'IDENTIFIER' and tok.value in cpp_keywords:
#        freqs[tok.value] += 1
#        print(tok)
#
#print(freqs)