
window.onload = function(){
	var a_backtop = document.getElementById('backtop');
	var fix_right = document.getElementById('fix_right');
	var istrue = true;
	var inter = null;
	var clientHeight = document.documentElement.clientHeight || document.body.clientHeight;

	window.onscroll = function(){
		/* 
		调用setInterval -> istrue = true -> onscroll -> !istrue(false) -> istrue = false -> 调用setInterval -> 循环
		当发生其他事件 : 调用setInterval -> istrue = true -> onscroll -> istrue = false -> [ 事件 ] -> onscroll -> !istrue(true) -> clearInterval(inter) 
		*/

		var a_height = document.documentElement.scrollTop || document.body.scrollTop;

		if (a_height >= (clientHeight-300)) {		/* 当滑动到第二屏幕的时候，自定出现 */
			a_backtop.style.display = 'block';	
			fix_right.style.display = 'block';
		}else{
			a_backtop.style.display = 'none';	
			fix_right.style.display = 'none';
		}

		if (!istrue){
			clearInterval(inter);
		}
		istrue = false;
	}

	a_backtop.onclick = function(){
		inter = setInterval(function(){
			var a_height = document.documentElement.scrollTop || document.body.scrollTop;
			var speed = Math.ceil(a_height/5+1);
			istrue = true;
			document.documentElement.scrollTop = document.body.scrollTop -= speed;
			if (Math.floor(a_height)==0){
				clearInterval(inter);
			}
		}, 20);
	}

}