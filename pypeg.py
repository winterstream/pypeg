#  Copyright (C) 2007 Chris Double.
#                2008 Wynand Winterbach - ported from Javascript to Python
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
# 
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
# 
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
# 
#  THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
#  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  DEVELOPERS AND CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import itertools as it
from collections import defaultdict

class ParseState(object):
    __slots__ = ['input', 'index', 'cache']
    parser_id = 0

    def __init__(self, input, index=None, cache=None):
        self.input = input
        self.index = index or 0
        self.cache = cache if cache else defaultdict(lambda: defaultdict(lambda: None))

    def fromP(self, index):
        return ParseState(self.input, self.index + index, self.cache)

    def substring(self, start, end):
        return self.input[start + self.index:end + self.index]

    def at(self, index):
        return self.input[self.index + index]

    def __repr__(self):
        return "ParseState(%s)" % self.input[self.index:]

    def getCached(self, pid):
        return self.cache[pid][self.index]
        
    def putCached(self, pid, val):
        self.cache[pid][self.index] = val
        return val

    def __len__(self):
        return len(self.input) - self.index

    @classmethod
    def getPid(cls):
        cls.parser_id += 1
        return cls.parser_id

ps = ParseState

class Result(object):
    __slots__ = ['remaining', 'matched', 'ast']
    
    def __init__(self, remaining, matched, ast):
        self.remaining = remaining
        self.matched = matched
        self.ast = ast

    def __repr__(self):
        return repr({'remaining': self.remaining,
                     'matched':   self.matched,
                     'ast':       self.ast})

def cacheable(f):
    def memoized_f(*args):
        pid = ParseState.getPid()
        parser = f(*args)

        def exec_and_memoize(state):
            cached = state.getCached(pid)
            return cached if cached else state.putCached(pid, parser(state))
        exec_and_memoize.func_name = f.func_name
        return exec_and_memoize
    return memoized_f

def asParser(p):
    return token(unicode(p)) if isinstance(p, (str, unicode)) else p

def convert_str_to_parser(f):
    def converted_f(*args):
        return f(*(asParser(arg) for arg in args))
    return converted_f

@cacheable
def token(s):
    def parser(state):
        if len(state) >= len(s) and state.substring(0, len(s)) == s:
            return Result(state.fromP(len(s)), s, s)
        else:
            return None
    return parser

@cacheable
def ch(c):
    def parser(state):
        if len(state) >= 1 and state.at(0) == c:
            return Result(state.fromP(1), c, c)
        else:
            return None
    return parser

@cacheable
def range_(lower, upper):
    def parser(state):
        if len(state) < 1:
            return None
        else:
            ch = state.at(0)
            if lower <= ch <= upper:
                return Result(state.fromP(1), ch, ch)
            else:
                return None
    return parser

@cacheable
@convert_str_to_parser
def action(p, f):
    def parser(state):
        x = p(state)
        if x:
            x.ast = f(x.ast)
            return x
        else:
            return None
    return parser

def join_action(p, sep):
    return action(asParser(p), lambda ast: sep.join(ast))

def left_factor(ast):
    return reduce(lambda v, action: [v, action], ast[1], ast[0])

@convert_str_to_parser
def left_factor_action(p):
    return action(p, left_factor)

@cacheable
@convert_str_to_parser
def negate(p):
    def parser(state):
        if len(state) > 1:
            if not p(state):
                return Result(state.fromP(1), state.at(0), state.at(0))
            else:
                return None
        else:
            return None
    return parser

def end_p(state):
    if len(state) == 0:
        return Result(state, None, None)
    else:
        return None

def nothing_p(state):
    return None

@cacheable
@convert_str_to_parser
def sequence(*parsers):
    def parser(state):
        ast = []
        matched = []

        for p in parsers:
            result = p(state)
            if result and result.ast:
                ast.append(result.ast)
                matched.append(result.matched)
            else:
                return None
            state = result.remaining
        return Result(state, u"".join(matched), ast)
    return parser

WHITESPACE_P = repeat0(choice(*(expect(ch(c)) for c in "\t\n\r ")))
@convert_str_to_parser
def whitespace(p):
    def parser(state):
        return p(WHITESPACE_P(state).remaining)
    return parser

@cacheable
@convert_str_to_parser
def wsequence(*parsers):
    return sequence(whitespace(p) for p in parsers)

@cacheable
@convert_str_to_parser
def choice(*parsers):
    def parser(state):
        for result in (p(state) for p in parsers):
            if result:
                return result
        return None
    return parser

@cacheable
@convert_str_to_parser
def butnot(p1, p2):
    def parser(state):
        ar, br = p1(state), p2(state)
        if not br:
            return ar
        else:
            if ar and (len(ar.matched) > len(br.matched)):
                return ar
            else:
                return None
    return parser

@cacheable
@convert_str_to_parser
def difference(p1, p2):
    def parser(state):
        ar, br = p1(state), p2(state)
        if not br:
            return ar
        else:
            if len(ar.matched) >= len(br.matched):
                return br
            else:
                return ar
    return parser

@cacheable
@convert_str_to_parser
def xor(p1, p2):
    def parser(state):
        ar, br = p1(state), p2(state)
        if ar and br:
            return None
        else:
            return ar or br
    return xor

def repeat_loop(p, state, result):
    ast = []
    matched = []

    while result:
        if result.ast != None:
            ast.append(result.ast)
            matched.append(result.matched)
        if result.remaining.index == state.index:
            break
        state  = result.remaining
        result = p(state)
    return Result(state, u"".join(matched), ast)

@cacheable
@convert_str_to_parser
def repeat0(p):
    def parser(state):
        return repeat_loop(p, state, p(state))
    return parser

@cacheable
@convert_str_to_parser
def repeat1(p):
    def parser(state):
        result = p(state)
        if not result:
            return None
        else:
            return repeat_loop(p, state, result)
    return parser

@cacheable
@convert_str_to_parser
def optional(p):
    def parser(state):
        return p(state) or Result(state, "", None)
    return parser

@convert_str_to_parser
def expect(p):
    return action(p, lambda ast: None)

@convert_str_to_parser
def chain(p, s, f):
    return action(sequence(p, repeat0(action(sequence(s, p), f))),
                  lambda ast: [ast[0]] + ast[1])

@convert_str_to_parser
def chainl(p, s):
    return action(sequence(p, repeat0(sequence(s, p))),
                  lambda ast: reduce(lambda v, action: action[0](v, action[1]), ast[1], ast[0]))

@convert_str_to_parser
def list_(p, s):
    return chain(p, s, lambda ast: ast[1])

@convert_str_to_parser
def wlist(*parsers):
    return list_(*(whitespace(p) for p in parsers))

def epsilon_p(state):
    return Result(state, u"", None)

@cacheable
def semantic(f):
    def parser(state):
        return Result(state, "", None) if f(state.ast) else None
    return parser

@cacheable
@convert_str_to_parser
def and_(p):
    def parser(state):
        return Result(state, u"", None) if p(state) else None
    return parser

@cacheable
@convert_str_to_parser
def not_(p):
    def parser(state):
        return None if p(state) else Result(state, u"", None)
    return parser

def flatmap(f, seq):
    work_stack = [iter(seq)]
    result = []

    while len(work_stack) > 0:
        try:
            item = work_stack[-1].next()
            if isinstance(item, (list, tuple)):
                work_stack.append(iter(item))
            else:
                result.append(f(item))
        except StopIteration:
            work_stack.pop()
    return result

def flatten(p):
    return action(p, lambda ast: flatmap(lambda x: x, ast))

def flat_join(p):
    return join_action(flatten(p), "")
