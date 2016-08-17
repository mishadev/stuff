# coding: utf-8

from __future__ import unicode_literals
import re
import nltk
from nltk.compat import htmlentitydefs
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
nltk.download("stopwords")
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      <3                         # heart
    )"""

# URL pattern due to John Gruber, modified by Tom Winzig. See
# https://gist.github.com/winzig/8894715

URLS = r"""			# Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}				# 1-3 slashes
      |					#   or
      [a-z0-9%]				# Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
    |					#   or
                                       # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:					# One or more:
    [^\s()<>{}\[\]]+			# Run of non-space, non-()<>{}[]
    |					#   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
  )+
  (?:					# End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
    |					#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
  	(?<!@)			        # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)			        # not succeeded by a @,
                            # avoid matching "foo.na" in "foo.na@example.com"
  )
"""

    # Phone numbers:
PHOMENUM = """
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?
      \d{3}          # exchange
      [\-\s.]*
      \d{4}          # base
    )"""


# The components of the tokenizer:
REGEXPS = (
    # ASCII Emoticons
    EMOTICONS
    ,
    # Remaining word types:
    r"""
    (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots.
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
    )

######################################################################
# This is the core tokenizing regex:

WORD_RE = re.compile(r"""(%s)""" % "|".join(REGEXPS), re.VERBOSE | re.I
                     | re.UNICODE)

# WORD_RE performs poorly on these patterns:
HANG_RE = re.compile(r'([^a-zA-Z0-9])\1{3,}')

# The emoticon string gets its own regex so that we can preserve case for
# them as needed:
EMOTICON_RE = re.compile(EMOTICONS, re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
ENT_RE = re.compile(r'&(#?(x?))([^&;\s]+);')

NONALPHABETIC_RE = re.compile(r"""^[^A-Za-z\(\)]$""", re.VERBOSE | re.I | re.UNICODE)

ARROW_RE = re.compile(r"""[\-]+>|<[\-]+""")

PHOMENUM_RE = re.compile(PHOMENUM, re.VERBOSE | re.I | re.UNICODE)
######################################################################
# Functions for converting html entities
######################################################################
def _str_to_unicode(text, encoding=None, errors='strict'):
    if encoding is None:
        encoding = 'utf-8'
    if isinstance(text, bytes):
        return text.decode(encoding, errors)
    return text

def _replace_html_entities(text, keep=(), remove_illegal=True, encoding='utf-8'):
    """
    Remove entities from text by converting them to their
    corresponding unicode character.

    :param text: a unicode string or a byte string encoded in the given
    `encoding` (which defaults to 'utf-8').

    :param list keep:  list of entity names which should not be replaced.\
    This supports both numeric entities (``&#nnnn;`` and ``&#hhhh;``)
    and named entities (such as ``&nbsp;`` or ``&gt;``).

    :param bool remove_illegal: If `True`, entities that can't be converted are\
    removed. Otherwise, entities that can't be converted are kept "as
    is".

    :returns: A unicode string with the entities removed.

    See https://github.com/scrapy/w3lib/blob/master/w3lib/html.py

        >>> from nltk.tokenize.casual import _replace_html_entities
        >>> _replace_html_entities(b'Price: &pound;100')
        'Price: \\xa3100'
        >>> print(_replace_html_entities(b'Price: &pound;100'))
        Price: £100
        >>>
    """

    def _convert_entity(match):
        entity_body = match.group(3)
        if match.group(1):
            try:
                if match.group(2):
                    number = int(entity_body, 16)
                else:
                    number = int(entity_body, 10)
                # Numeric character references in the 80-9F range are typically
                # interpreted by browsers as representing the characters mapped
                # to bytes 80-9F in the Windows-1252 encoding. For more info
                # see: http://en.wikipedia.org/wiki/Character_encodings_in_HTML
                if 0x80 <= number <= 0x9f:
                    return int2byte(number).decode('cp1252')
            except ValueError:
                number = None
        else:
            if entity_body in keep:
                return match.group(0)
            else:
                number = htmlentitydefs.name2codepoint.get(entity_body)
        if number is not None:
            try:
                return unichr(number)
            except ValueError:
                pass

        return "" if remove_illegal else match.group(0)

    return ENT_RE.sub(_convert_entity, _str_to_unicode(text, encoding))


######################################################################
class TweetTokenizerPlus(TweetTokenizer):
    def __init__(self, preserve_case=True,
                strip_handles=True, reduce_len=True):
        TweetTokenizer.__init__(self, preserve_case,
                strip_handles, reduce_len)

    def _remove_links(self, text):
        return re.compile(URLS, re.I | re.UNICODE | re.VERBOSE).sub("", text)

    def _remove_stopwords(self, words):
        stop = set(stopwords.words('english') + ['RT'])
        return [word for word in words if word not in stop]

    def _remove_nonalphabetic(self, words):
        return [word for word in words if not NONALPHABETIC_RE.search(word)]

    def _remove_phonenumbers(self, text):
        return PHOMENUM_RE.sub("", text)

    def _remove_arrows(self, text):
        return ARROW_RE.sub("", text)

    def tokenize(self, text):
        # Fix HTML character entities:
        text = _replace_html_entities(text)
        # Remove Links from text
        text = self._remove_links(text)
        text = self._remove_phonenumbers(text)
        text = self._remove_arrows(text)
        # Remove username handles
        if self.strip_handles:
            text = self._remove_handles(text)
        # Normalize word lengthening
        if self.reduce_len:
            text = self._reduce_lengthening(text)
        # Shorten problematic sequences of characters
        safe_text = HANG_RE.sub(r'\1\1\1', text)
        # Tokenize:
        words = WORD_RE.findall(safe_text)
        # Possibly alter the case, but avoid changing emoticons like :D into :d:
        #text = self._remove_nonalphabetic(safe_text)
        if not self.preserve_case:
            words = list(map((lambda x : x if EMOTICON_RE.search(x) else
                              x.lower()), words))
        words = self._remove_stopwords(words)
        words = self._remove_nonalphabetic(words)
        return words

######################################################################
# Normalization Functions
######################################################################

    def _reduce_lengthening(self, text):
        """
        Replace repeated character sequences of length 3 or greater with sequences
        of length 3.
        """
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)

    def _remove_handles(self, text):
        """
        Remove Twitter username handles from text.
        """
        pattern = re.compile(r"(^|(?<=[^\w.-]))@[A-Za-z_]+\w+", re.I | re.UNICODE | re.VERBOSE)
        return pattern.sub('', text)

######################################################################
# Tokenization Function
######################################################################

def casual_tokenize(text, preserve_case=True, reduce_len=False, strip_handles=False):
    """
    Convenience function for wrapping the tokenizer.
    """
    return TweetTokenizer(preserve_case=preserve_case, reduce_len=reduce_len,
                          strip_handles=strip_handles).tokenize(text)

###############################################################################
