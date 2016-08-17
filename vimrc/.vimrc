let g:solarized_termcolors=256
let g:solarized_bold=1
let g:solarized_italic=1
let g:solarized_underline=1
let g:solarized_contrast="high"
colorscheme solarized

let g:indent_guides_auto_colors = 0
hi Visual ctermfg=14 ctermbg=15
hi Search ctermfg=1 ctermbg=15
hi MatchTag ctermfg=15 ctermbg=5
hi MatchParen ctermfg=15 ctermbg=5

nmap <C-z> :w<CR>
imap <C-z> <ESC>:w<CR>
nmap <leader>pp viw"0p
nmap <CR> o<Esc>k
nmap <S-CR> O<Esc>j
nmap <leader><space> a<space><Esc>
nnoremap * :keepjumps normal! mf*`f<CR>
command Q q
