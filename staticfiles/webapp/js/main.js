/* show file value after file select */
document.querySelector(".custom-file-input").addEventListener("change", function (e) {
  var fileName = document.getElementById("imageFile").files[0].name;
  var nextSibling = e.target.nextElementSibling;
  nextSibling.innerText = fileName;
});

$(document).ready(function () {
  $("#myTable").DataTable({
    searching: false,
    paging: false,
    info: false,

    order: [[1, "desc"]],
    columnDefs: [
      {
        orderable: false,
        targets: 0,
      },
    ],
  });
});
