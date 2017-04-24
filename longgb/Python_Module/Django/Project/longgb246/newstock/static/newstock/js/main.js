$(document).ready(function(){

	//slider init
	$("#slides").responsiveSlides({
		speed: 700,     //Speed of the transition
		timeout: 3500,  //Time between slide transitions
		pager: true     //Show the pagination
	});
	

	//wow.js init
	wow = new WOW(
	    {
		  animateClass: 'animated',
		  mobile: false,
		  offset: 100
		}
	);
	wow.init();
	
});





