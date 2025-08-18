document.getElementById('download-btn').addEventListener('click', () => {
    const { jsPDF } = window.jspdf;
    const certificate = document.getElementById('certificate-to-download');
    const downloadButton = document.getElementById('download-btn');
    const linkElement = document.getElementById('pdf-link');

    // Get position of link relative to the certificate container
    const scale = 3; // This should match the scale in html2canvas
    const certificateRect = certificate.getBoundingClientRect();
    const linkRect = linkElement.getBoundingClientRect();
    
    // Calculate the scaled coordinates for the PDF link
    const pdfX = (linkRect.left - certificateRect.left) * scale;
    const pdfY = (linkRect.top - certificateRect.top) * scale;
    const pdfWidth = linkRect.width * scale;
    const pdfHeight = linkRect.height * scale;

    // Hide the button before taking the screenshot
    downloadButton.style.display = 'none';

    // Use html2canvas to render the certificate div as a canvas
    html2canvas(certificate, {
        scale: scale, // Increase scale for better resolution
        useCORS: true,
        backgroundColor: '#ffffff'
    }).then(canvas => {
        // Get image data from canvas as JPEG for smaller file size
        const imgData = canvas.toDataURL('image/jpeg', 0.9); // 0.9 is quality
        
        // Determine PDF orientation based on the certificate's aspect ratio
        const orientation = canvas.width > canvas.height ? 'l' : 'p';
        const pdf = new jsPDF(orientation, 'px', [canvas.width, canvas.height]);

        const pdfWidth = pdf.internal.pageSize.getWidth();
        const pdfHeight = pdf.internal.pageSize.getHeight();

        // Add the image to the PDF
        pdf.addImage(imgData, 'JPEG', 0, 0, pdfWidth, pdfHeight);
        
        // Add the clickable link to the PDF
        pdf.link(pdfX, pdfY, pdfWidth, pdfHeight, { url: 'https://pinecoder.in' });

        // Download the PDF
        pdf.save('Support-Certificate-Pinecoder.pdf');

        // Show the button again after the download
        downloadButton.style.display = 'block';
    }).catch(err => {
        console.error("Error generating PDF:", err);
        // Ensure the button is shown again even if there's an error
        downloadButton.style.display = 'block';
    });
});