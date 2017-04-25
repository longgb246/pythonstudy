" Basic setting
set ignorecase
hi Search guibg=peru guifg=wheat

:set hlsearch

"R and python and pig
set nocompatible
syntax on
syntax enable
filetype plugin on
filetype indent on

" R disable replace _ to <-
set <M-->=^[-
let vimrplugin_assign_map = "<M-->"

" Python tab
let OPTION_NAME = 1
let python_highlight_all = 1

set tabstop=8 
set expandtab 
set shiftwidth=4 
set softtabstop=4

set background=dark

" For the pig plugin
augroup filetypedetect
au BufNewFile,BufRead *.pig set filetype=pig syntax=pig
augroup End

" For the md plugin
augroup filetypedetect
au BufNewFile,BufRead *.md set filetype=markdown syntax=markdown
augroup End


"The colors scheme
colorscheme vividchalk

" Lines added by the Vim-R-plugin command :RpluginConfig (2014-Dec-19 00:56):
" Press the space bar to send lines (in Normal mode) and selections to R:
vmap <Space> <Plug>RDSendSelection
nmap <Space> <Plug>RDSendLine

" Force Vim to use 256 colors if running in a capable terminal emulator:
if &term =~ "xterm" || &term =~ "256" || $DISPLAY != "" || $HAS_256_COLORS == "yes"
    set t_Co=256
endif

" Lines added by the Vim-R-plugin command :RpluginConfig (2014-Dec-19 01:05):
