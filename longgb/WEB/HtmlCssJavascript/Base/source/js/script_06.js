window.onload = function() {
    /* 1、click点击tabs */
    var oTab = document.getElementById("tabs_click_01");
    var oU1 = oTab.getElementsByTagName("ul")[0];
    var oLis = oU1.getElementsByTagName("li");
    var oDivs = oTab.getElementsByTagName('div');
    for (var i=0, len = oLis.length; i<len; i++){
        oLis[i].index = i;
        oLis[i].onclick = function(){
            for (var n=0; n<len; n++){
                oLis[n].className = "";
                oDivs[n].className = "hide";
            }
            this.className = "on";
            oDivs[this.index].className = "";
        }
    }

    /* 2、移动的时候，自动切换 */
    var move_auto_tab = document.getElementById('tabs_hover');
    var move_auto_li = move_auto_tab.getElementsByTagName('li');    
    var move_auto_div = move_auto_tab.getElementsByTagName('div');    
    var timer = null;
    var this_index = 0;
    var global_timer = null;

    function onePlay(auto_li, auto_div, now_index){
        for (var i=0; i<auto_li.length; i++){
            auto_li[i].setAttribute("class", "");
            auto_div[i].setAttribute("class", "hide");
        }
        auto_li[now_index].setAttribute("class", "on");
        auto_div[now_index].setAttribute("class", "");
    };

    function autoPlay(auto_li, auto_div){
        if (this_index < 4){
            this_index++;    
        }else{
            this_index = 0;    
        };
        onePlay(auto_li, auto_div, this_index);
    }

    for (var i=0; i<move_auto_li.length; i++){
        move_auto_li[i].index = i
        move_auto_li[i].onmouseover = function(){   /* 开启则为滑动切换 */
        // move_auto_li[i].onclick = function(){    /* 开启则为点击切换 */
            var that = this;
            if (timer){                             /* 计时器设置延时划过 */
                clearTimeout(timer);
                timer = null;
            };
            if (global_timer){
                clearTimeout(global_timer);
                global_timer = null;  
            }
            timer = setTimeout(
            onePlay(move_auto_li, move_auto_div, that.index)
            , 0);                                  /* 设置延时划过时间 */
        };  
        move_auto_li[i].onmouseout = function(){   /* 鼠标移出，开启自动脚本 */
            var that = this;
            this_index = that.index;
            global_timer = setInterval(
                function(){
                    autoPlay(move_auto_li, move_auto_div)
                }, 2000)
            }
    }

    global_timer = setInterval(
        function(){
            autoPlay(move_auto_li, move_auto_div)
        }, 2000)
   

    
    /* 3、图片轮播 */
    var imgs_div = document.getElementById('imgCarousel');
    var imgs_li = imgs_div.getElementsByTagName('li');
    var imgs = imgs_div.getElementsByTagName('img');
    var imgs_global_timer = null;
    var imgs_index = 0;

    function imgs_autoPlay(auto_li, auto_div){
        if (imgs_index < 4){
            imgs_index++;    
        }else{
            imgs_index = 0;    
        };
        onePlay(auto_li, auto_div, imgs_index);
    }

    for (var i=0; i<imgs_li.length; i++){
        imgs_li[i].index = i
        imgs_li[i].onmouseover = function(){   
            var that = this;
            if (imgs_global_timer){
                clearTimeout(imgs_global_timer);
                imgs_global_timer = null;  
            }
            onePlay(imgs_li, imgs, that.index);
        }
        imgs_li[i].onmouseout = function(){   /* 鼠标移出，开启自动脚本 */
            var that = this;
            imgs_index = that.index;
            imgs_global_timer = setInterval(
                function(){
                    imgs_autoPlay(imgs_li, imgs)
                }, 4000)
        }
    }

    imgs_global_timer = setInterval(
        function(){
            imgs_autoPlay(imgs_li, imgs)
        }, 4000)

}          

// 字体： Open Sans Condensed