console.log("hopefully this will work ABCDEF")

// // this is a comment
//
// $(document).ready(function () {
//     console.log('web page is ready LOG FILE');
//     // # symbol is 'id' in html file
//     $('#inference').click(async function () {
//         // aynsc is what happens at the same time when you click button
//         console.log('button was clicked');
//         const cylinders = parseFloat($('#cylinders').val());
//         const horsepower = parseFloat($('#horsepower').val());
//         const weight = parseFloat($('#weight').val());
//         const data = {
//             // for JS dicts, you don't have to put both key-value
//             cylinders,
//             horsepower,
//             weight
//         };
//         console.log(data);
//
//         // talks to python script server to run inference()
//         const response = await $.ajax('/inference', {
//             data: JSON.stringify(data),
//             method: 'post',
//             contentType: 'application/json'
//         });
//
//         console.log('response', response);
//         $('#mpg').val(response.predictions);
//     });
//     $('#scatter-button').click(async function () {
//         console.log('scatter');
//         const response = await $.ajax('/plot');
//         const mpg = response.map(a => a[0]);
//         const weight = response.map(a => a[1]);
//         console.log(mpg);
//         const trace = [{
//             x: weight,
//             y: mpg,
//             mode: 'markers',
//             type: 'scatter'
//         }];
//         const layout = {
//             xaxis: {
//                 title: 'Weight'
//             },
//             yaxis: {
//                 title: 'MPG'
//             },
//             title: 'MPG by Weight',
//             width: 700,
//             height: 300
//         };
//         Plotly.plot($('#graph1')[0], trace, layout);
//     });
//     $('#histo-button').click(async function () {
//         console.log('histogram');
//     });
// });
