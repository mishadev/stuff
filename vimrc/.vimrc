let g:solarized_termcolors=256
let g:solarized_bold=1
let g:solarized_italic=1
let g:solarized_underline=1
let g:solarized_contrast="high"
colorscheme solarized

let g:indent_guides_auto_colors = 0
hi Visual ctermfg=14 ctermbg=15
hi Search ctermfg=5 ctermbg=15
hi MatchTag ctermfg=15 ctermbg=5
hi MatchParen ctermfg=15 ctermbg=5

map <C-z> :w<CR>
nmap <leader>ww ciw()<ESC>P
