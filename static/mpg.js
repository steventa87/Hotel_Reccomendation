$(document).ready(function () {
    console.log('web page is ready' + new Date());
    // # symbol is 'id' in html file
    $('#search').click(async function () {
        // aynsc is what happens at the same time when you click button
        console.log('button was clicked');
        const search = $('#searchValue').val();
        const data = {
            // for JS dicts, you don't have to put both key-value
            search
        };
        console.log(data);

        // talks to python script server to run inference()
        const response = await $.ajax('/search', {
            data: JSON.stringify(data),
            method: 'post',
            contentType: 'application/json'
        });
        console.log("this is ajax response string: ", response)

        let hotel1 = response.hotel1;
        let hotel2 = response.hotel2;
        let hotel3 = response.hotel3;

        let search1= hotel1.split(' ').join('+')
        let search2= hotel2.split(' ').join('+')
        let search3= hotel3.split(' ').join('+')

        let review1 = response.review1;
        let review2 = response.review2;
        let review3 = response.review3;

        // let progressBar1 = response.progressBar1
        // let progressBar2 = response.progressBar2
        // let progressBar3 = response.progressBar3

        $('#hotelRec').empty();

        $('#hotelRec').append('<div class="col-md-4 padding-percentage">'+
            '<div class="hotelBoxes whiteBackground" id="hotel1">'+
                '<a href="http://www.google.com/search?q='+search1+'" target=_blank>'+hotel1+'</a>'+
                '<hr/>'+
                '<p>'+ review1 +'<p>'+
                '<hr/>'+
                '<button type="button" class="btn btn-outline-danger btn-block" id="topicModel1">Topic Model</button>'+
                // progresbar1 stuff
                // '<div class="progress" style="height: 20px;">' +
                //     '<div class="progress-bar bg-danger" role="progressbar" style="width:'+progressBar1+'%;"' + 'aria-valuenow='+progressBar1+ 'aria-valuemin="0" aria-valuemax="100">'+progressBar1+'</div>'+
                // '</div>'+

            '</div>'+

        '</div>'+

        '<div class="col-md-4 padding-percentage">'+
            '<div class="hotelBoxes whiteBackground" id="hotel2">'+
                '<a href="http://www.google.com/search?q='+search2+'" target=_blank>'+hotel2+'</a>'+
                '<hr/>'+
                '<p>'+ review2 +'</p>'+
                '<hr/>'+
                '<button type="button" class="btn btn-outline-danger btn-block" id="topicModel2">Topic Model</button>'+
                //progressbar2 stuff
                // '<div class="progress" style="height: 20px;">' +
                //     '<div class="progress-bar bg-danger" role="progressbar" style="width:'+progressBar2+'%;"' + 'aria-valuenow='+progressBar2+ 'aria-valuemin="0" aria-valuemax="100">'+progressBar2+'</div>'+
                // '</div>'+

            '</div>'+
        '</div>'+

        '<div class="col-md-4 padding-percentage">'+
            '<div class="hotelBoxes whiteBackground" id="hotel3">'+
                '<a href="http://www.google.com/search?q='+search3+'" target=_blank>'+hotel3+'</a>'+
                '<hr/>'+
                '<p>'+review3 +'</p>'+
                '<hr/>'+
                '<button type="button" class="btn btn-outline-danger btn-block" id="topicModel3">Topic Model</button>'+
                // progressbar3 stuff
                // '<div class="progress" style="height: 20px;">' +
                //     '<div class="progress-bar bg-danger" role="progressbar" style="width:'+progressBar3+'%;"' + 'aria-valuenow='+progressBar3+ 'aria-valuemin="0" aria-valuemax="100">'+progressBar3+'</div>'+
                // '</div>'+

            '</div>'+
        '</div>')

        $("#topicModel1").click(function(){
            $("#myModal").modal();
        });

        $("#topicModel2").click(function(){
            $("#myModal").modal();
        });

        $("#topicModel3").click(function(){
            $("#myModal").modal();
        });

    });
});
// old code
// $(document).ready(function () {
//     console.log('web page is ready' + new Date());
//     // # symbol is 'id' in html file
//     $('#search').click(async function () {
//         // aynsc is what happens at the same time when you click button
//         console.log('button was clicked');
//         const search = $('#searchValue').val();
//         const data = {
//             // for JS dicts, you don't have to put both key-value
//             search
//         };
//         console.log(data);
//
//         // talks to python script server to run inference()
//         const response = await $.ajax('/search', {
//             data: JSON.stringify(data),
//             method: 'post',
//             contentType: 'application/json'
//         });
//         console.log("this is ajax response string: ", response)
//
//         let hotel1 = response.hotel1;
//         let hotel2 = response.hotel2;
//         let hotel3 = response.hotel3;
//
//         let review1 = response.review1;
//         let review2 = response.review2;
//         let review3 = response.review3;
//         // console.log('response', response);
//         $('#hotelRec').empty();
//
//         $('#hotelRec').append('<div class="col-md-4 padding-percentage">'+
//             '<div class="hotelBoxes whiteBackground" id="hotel1">'+
//                 '<h6> '+ hotel1+' </h6>'+
//                 '<hr/>'+
//                 '<p>'+ review1 +'<p>'+
//             '</div>'+
//         '</div>'+
//
//         '<div class="col-md-4 padding-percentage">'+
//             '<div class="hotelBoxes whiteBackground" id="hotel2">'+
//                 '<h6> '+ hotel2+' </h6>'+
//                 '<hr/>'+
//                 '<p>'+ review2 +'</p>'+
//             '</div>'+
//         '</div>'+
//
//         '<div class="col-md-4 padding-percentage">'+
//             '<div class="hotelBoxes whiteBackground" id="hotel3">'+
//                 '<h6> '+ hotel3 +' </h6>'+
//                 '<hr/>'+
//                 '<p>'+review3 +'</p>'+
//             '</div>'+
//         '</div>')
//         // copy code here
//         $("#hotel1").click(function(){
//                 $("#myModal").modal();
//             });
//
//             $("#hotel2").click(function(){
//                 $("#myModal").modal();
//             });
//
//             $("#hotel3").click(function(){
//                 $("#myModal").modal();
//             });
//         // end copy
//     });
// });
