// Function to create a new trigger to run the task every 5 minutes
function startTrigger() {
    ScriptApp.newTrigger("createGoogleFormsWithMetadata")
      .timeBased()
      .everyMinutes(10)  // Change to a longer interval if needed to avoid overlap
      .create();
  }
  
  // Function to delete the trigger that runs the task
  function deleteTriggers() {
    const triggers = ScriptApp.getProjectTriggers();
    for (const trigger of triggers) {
      if (trigger.getHandlerFunction() === "createGoogleFormsWithMetadata") {
        ScriptApp.deleteTrigger(trigger);
      }
    }
  }
  
  // Function that will be triggered every 5 minutes
  function createGoogleFormsWithMetadata() {
    // Get the script properties to track progress
    const scriptProperties = PropertiesService.getScriptProperties();
    
    // Retrieve the set of processed form IDs (if any)
    let processedFormIds = scriptProperties.getProperty('processedFormIds');
    if (processedFormIds) {
      processedFormIds = JSON.parse(processedFormIds); // Parse it as a JSON array
    } else {
      processedFormIds = []; // Initialize it as an empty array
    }
    
    // Replace with your folder ID
    const folderId = '1x7G4ro56dgYVf79SVXtBF2GbSIqPJSbQ'; // Replace with your folder ID
    const folder = DriveApp.getFolderById(folderId);
    
    // Get all Google Forms in the folder
    const files = folder.getFilesByType(MimeType.GOOGLE_FORMS);
    
    // Open the active spreadsheet and get or create the sheet for storing responses
    const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    let newSheet = spreadsheet.getSheetByName('Form Responses');
    if (!newSheet) {
      newSheet = spreadsheet.insertSheet('Form Responses');
    }
    
    // Dynamically create the header row based on the number of questions
    const firstForm = files.next();  // Get the first file to inspect its questions
    const firstFormObj = FormApp.openById(firstForm.getId());
    const firstFormItems = firstFormObj.getItems();
    
    let headerRow = ['Form Name', 'Response ID'];
    firstFormItems.forEach(function(item, index) {
      headerRow.push('Question ' + (index + 1), 'Answer ' + (index + 1));
    });
  
    // Set up the header row in the new sheet if it's empty
    if (newSheet.getLastRow() === 0) {
      newSheet.appendRow(headerRow);
    }
  
    // Iterate through each form in the folder
    while (files.hasNext()) {
      const file = files.next(); // Get the next file (form)
      
      const formId = file.getId(); // Get the Form ID
      console.log(processedFormIds.length)
      if (processedFormIds.includes(formId)) {
        continue; // Skip if this form was already processed
      }
  
      const form = FormApp.openById(formId); // Open the form using the Form ID
      
      // Get the form's responses
      const responses = form.getResponses();
      
      // Iterate over each response for the current form
      responses.forEach(function(response) {
        const responseId = response.getId(); // Get response ID
        const itemResponses = response.getItemResponses();
        
        // Create an array to hold the form data (Form Name, Response ID, Questions and Answers)
        const rowData = [file.getName(), responseId];
        
        // Add each question and answer to the rowData array
        itemResponses.forEach(function(itemResponse, index) {
          const question = itemResponse.getItem().getTitle();
          const answer = itemResponse.getResponse();
          rowData.push(question, answer);
        });
        
        // Append the row data (Form Name, Response ID, Questions and Answers) to the new sheet
        newSheet.appendRow(rowData);
      });
      
      // After processing each form, add its form ID to the set of processed IDs
      processedFormIds.push(formId);
      
      // Save the updated set of processed form IDs back to Script Properties
      scriptProperties.setProperty('processedFormIds', JSON.stringify(processedFormIds));
    }
    
    Logger.log("Google Forms with metadata processed successfully.");
  }
  