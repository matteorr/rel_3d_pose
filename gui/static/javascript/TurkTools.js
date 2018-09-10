//credit to http://timbrady.org/turk/TimTurkTools.js

function GetWorkerId() {
	var workerId = turkGetParam( 'workerId', 'NONE' );
	return workerId;
}	

function IsTurkPreview() {
	return GetWorkerId() == "NONE";
}	

function IsOnTurk() {
  try {
    return ((window.location.host.indexOf('mturk')!=-1) || document.forms["mturk_form"] ||
      (top != self && window.parent.location.host.indexOf("mturk") != -1));
  } catch(err) {
    // insecure content trying to access https://turk probably, so say yes:
    return true;
  }
}

function GetAssignmentId() {
	var assignmentId = turkGetParam( 'assignmentId', 'NONE' );
	return assignmentId;
}	

// Check if this worker's actual ID is on the block list or not
function CheckBlockList(blockListName, funcForBlocked) {
	if (!IsTurkPreview()) {
		$.ajax({             
			crossDomain: true,
			dataType: 'jsonp',	
			url: 'https://timbrady.org/turk/checkDuplicates.pl', 
			data:{'workerId': GetWorkerId(), 
			 'blockListName': blockListName}, 
			success: function(data) {
				$('#serverContacted').val(1);
				if (data.blocked == 1) {
					funcForBlocked();
				}
			},
			error: function(jx, status, error) {
				$('#serverContacted').val(0);
			}
		});
	}
}

function IsNewWorker( _amt_worker_id ) {

	var _worker_exp = 0;
	$.ajax({
		url: "/mturk/IsNewWorker/",
		type: "POST",
		contentType: "application/json",
		data: JSON.stringify({
			"_worker_id": _amt_worker_id,
		}),
		dataType: "text",
		success: function( returned_data ) {
			console.debug( "   $> IsNewWorker() success _worker_exp: " + returned_data );
			if ( returned_data != '0' ){
				_worker_exp = parseInt(returned_data);
			}
		}
	});
	return _worker_exp;
}

function IsBlockedWorker( _amt_worker_id ) {

	var _is_blocked = false;
	$.ajax({
		url: "/mturk/IsBlockedWorker/",
		type: "POST",
		contentType: "application/json",
		data: JSON.stringify({
			"_worker_id": _amt_worker_id,
		}),
		dataType: "text",
		success: function( returned_data ) {
			console.debug( "   $> IsBlockedWorker() success:" + returned_data );
			if ( returned_data == 'REJECT' ){
				_is_blocked = true;
			}
		}
	});
	return _is_blocked;
}

function RandomOrder(num) {
  var order = new Array();
  for (var i=0; i<num; i++) { 
    order.push(i); 
  }
  return Shuffle(order);
}

Shuffle = function(o) { 
	for(var j, x, i = o.length; i; j = parseInt(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
	return o;
};

//For todays date;
Date.prototype.today = function(){ 
    return ((this.getDate() < 10)?"0":"") + this.getDate() +"/"+(((this.getMonth()+1) < 10)?"0":"") + (this.getMonth()+1) +"/"+ this.getFullYear() 
};
//For the time now
Date.prototype.timeNow = function(){
     return ((this.getHours() < 10)?"0":"") + this.getHours() +":"+ ((this.getMinutes() < 10)?"0":"") + this.getMinutes() +":"+ ((this.getSeconds() < 10)?"0":"") + this.getSeconds();
};

/**
 * Gets a URL parameter from the query string
 */
function turkGetParam( name, defaultValue ) { 
   var regexS = "[\?&]"+name+"=([^&#]*)"; 
   var regex = new RegExp( regexS ); 
   var tmpURL = window.location.href; 
   var results = regex.exec( tmpURL ); 
   if( results == null ) { 
     return defaultValue; 
   } else { 
     return results[1];    
   } 
}

/**
 * URL decode a parameter
 */
function decode(strToDecode)
{
  var encoded = strToDecode;
  return unescape(encoded.replace(/\+/g,  " "));
}


/**
 * Returns the Mechanical Turk Site to post the HIT to (sandbox. prod)
 */
function turkGetSubmitToHost() {
  return decode(turkGetParam("turkSubmitTo", "https://www.mturk.com"));
}


/**
 * Sets the assignment ID in the form. Defaults to use mturk_form and submitButton
 */ 
function turkSetAssignmentID( form_name, button_name ) {

  if (form_name == null) {
    form_name = "mturk_form";
  }

  if (button_name == null) {
    button_name = "submitButton";
  }

  assignmentID = turkGetParam('assignmentId', "");
  document.getElementById('assignmentId').value = assignmentID;

  if (assignmentID == "ASSIGNMENT_ID_NOT_AVAILABLE") { 
    // If we're previewing, disable the button and give it a helpful message
    btn = document.getElementById(button_name);
    if (btn) {
      btn.disabled = true; 
      btn.value = "You must ACCEPT the HIT before you can submit the results.";
    } 
  }

  form = document.getElementById(form_name); 
  if (form) {
     form.action = turkGetSubmitToHost() + "/mturk/externalSubmit"; 
  }
}



/*
 * jQuery Hotkeys Plugin
 * Copyright 2010, John Resig
 * Dual licensed under the MIT or GPL Version 2 licenses.
 *
 * Based upon the plugin by Tzury Bar Yochay:
 * http://github.com/tzuryby/hotkeys
 *
 * Original idea by:
 * Binny V A, http://www.openjs.com/scripts/events/keyboard_shortcuts/
*/

(function(jQuery){

	jQuery.hotkeys = {
		version: "0.8",

		specialKeys: {
			8: "backspace", 9: "tab", 13: "return", 16: "shift", 17: "ctrl", 18: "alt", 19: "pause",
			20: "capslock", 27: "esc", 32: "space", 33: "pageup", 34: "pagedown", 35: "end", 36: "home",
			37: "left", 38: "up", 39: "right", 40: "down", 45: "insert", 46: "del", 
			96: "0", 97: "1", 98: "2", 99: "3", 100: "4", 101: "5", 102: "6", 103: "7",
			104: "8", 105: "9", 106: "*", 107: "+", 109: "-", 110: ".", 111 : "/", 
			112: "f1", 113: "f2", 114: "f3", 115: "f4", 116: "f5", 117: "f6", 118: "f7", 119: "f8", 
			120: "f9", 121: "f10", 122: "f11", 123: "f12", 144: "numlock", 145: "scroll", 191: "/", 224: "meta"
		},

		shiftNums: {
			"`": "~", "1": "!", "2": "@", "3": "#", "4": "$", "5": "%", "6": "^", "7": "&", 
			"8": "*", "9": "(", "0": ")", "-": "_", "=": "+", ";": ": ", "'": "\"", ",": "<", 
			".": ">",  "/": "?",  "\\": "|"
		}
	};

	function keyHandler( handleObj ) {
		// Only care when a possible input has been specified
		if ( typeof handleObj.data !== "string" ) {
			return;
		}

		var origHandler = handleObj.handler,
			keys = handleObj.data.toLowerCase().split(" ");

		handleObj.handler = function( event ) {
			// Don't fire in text-accepting inputs that we didn't directly bind to
			if ( this !== event.target && (/textarea|select/i.test( event.target.nodeName ) ||
				 event.target.type === "text") ) {
				return;
			}

			// Keypress represents characters, not special keys
			var special = event.type !== "keypress" && jQuery.hotkeys.specialKeys[ event.which ],
				character = String.fromCharCode( event.which ).toLowerCase(),
				key, modif = "", possible = {};

			// check combinations (alt|ctrl|shift+anything)
			if ( event.altKey && special !== "alt" ) {
				modif += "alt+";
			}

			if ( event.ctrlKey && special !== "ctrl" ) {
				modif += "ctrl+";
			}

			// TODO: Need to make sure this works consistently across platforms
			if ( event.metaKey && !event.ctrlKey && special !== "meta" ) {
				modif += "meta+";
			}

			if ( event.shiftKey && special !== "shift" ) {
				modif += "shift+";
			}

			if ( special ) {
				possible[ modif + special ] = true;

			} else {
				possible[ modif + character ] = true;
				possible[ modif + jQuery.hotkeys.shiftNums[ character ] ] = true;

				// "$" can be triggered as "Shift+4" or "Shift+$" or just "$"
				if ( modif === "shift+" ) {
					possible[ jQuery.hotkeys.shiftNums[ character ] ] = true;
				}
			}

			for ( var i = 0, l = keys.length; i < l; i++ ) {
				if ( possible[ keys[i] ] ) {
					return origHandler.apply( this, arguments );
				}
			}
		};
	}

	jQuery.each([ "keydown", "keyup", "keypress" ], function() {
		jQuery.event.special[ this ] = { add: keyHandler };
	});

})( jQuery );
