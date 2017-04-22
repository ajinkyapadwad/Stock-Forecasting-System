/*
* Search.js
* This is for menubar search result
*/

$(function(){
	var url = $('#typeahead').attr('data-url');
	console.log(url);
	// var usr_url = $('#typeahead').attr('usr-url');

	var entityname = new Bloodhound({
		datumTokenizer: Bloodhound.tokenizers.obj.whitespace('name'),
		queryTokenizer: Bloodhound.tokenizers.whitespace,
		limit: 5,
		prefetch: {
			url: url,
			ttl : 500,
  		}
	});


	entityname.initialize();
	console.log(entityname);
	// username.initialize();
 
	$('#search #typeahead').typeahead({
			highlight : true
		},
		{
			name: 'entity',
			displayKey: 'name',
			source: entityname.ttAdapter(),
			templates: {
			    header: '<h4 class="search-title"></h4>',
			    suggestion: Handlebars.compile('<p class="sugg-title"><strong>{{name}}</strong></p>'),
			    // empty: '<p><a href="/quote/add/" class="empty-search"></a></p>'
			}
		});

	// This is new infrastructure for add Entity direct from searchbar
	// $('.typeahead').keyup(function(event) {
	// 	$this = $( this );
	// 	$('.empty-search').text( $this.val() );
	// });
	

	$('.typeahead').on('typeahead:autocompleted', function (e, datum) {
		console.log(datum);
		window.location.href = '/search/' + datum.slug + '/';			
	});

	$('.typeahead').on('typeahead:selected', function (e, datum) {
		window.location.href = '/search/' + datum.slug + '/';
	});

});