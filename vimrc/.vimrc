set updatetime=250
let g:solarized_termcolors=256
let g:solarized_bold=1
let g:solarized_italic=1
let g:solarized_underline=1
let g:solarized_contrast="high"
colorscheme solarized

let g:indent_guides_auto_colors = 0
hi Visual ctermfg=14 ctermbg=15
hi Search ctermfg=8 ctermbg=16
hi MatchTag ctermfg=15 ctermbg=5
hi MatchParen ctermfg=15 ctermbg=5
hi SignColumn ctermbg=2
hi ExtraWhitespace ctermbg=red

augroup WhitespaceMatch
  " Remove ALL autocommands for the WhitespaceMatch group.
  autocmd!
  autocmd BufWinEnter * let w:whitespace_match_number =
       \ matchadd('ExtraWhitespace', '\s\+$')
  autocmd InsertEnter * call s:ToggleWhitespaceMatch('i')
  autocmd InsertLeave * call s:ToggleWhitespaceMatch('n')
augroup END
function! s:ToggleWhitespaceMatch(mode)
  let pattern = (a:mode == 'i') ? '\s\+\%#\@<!$' : '\s\+$'
  if exists('w:whitespace_match_number')
    call matchdelete(w:whitespace_match_number)
    call matchadd('ExtraWhitespace', pattern, 10, w:whitespace_match_number)
  else
    " Something went wrong, try to be graceful.
    let w:whitespace_match_number =  matchadd('ExtraWhitespace', pattern)
  endif
endfunction

nmap <F8> :TagbarOpenAutoClose<CR>
nmap <C-z> :w<CR>
imap <C-z> <ESC>:w<CR>
nmap <leader>pp viw"0p
nmap <CR> o<Esc>k
nmap <S-CR> O<Esc>j
nmap <leader><space> a<space><Esc>
nnoremap * :keepjumps normal! mf*`f<CR>
command Q q
