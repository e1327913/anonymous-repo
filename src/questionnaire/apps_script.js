function startTrigger() {
  ScriptApp.newTrigger("createGoogleFormsWithMetadata")
    .timeBased()
    .everyMinutes(5)
    .create();
}

function deleteTriggers() {
  const triggers = ScriptApp.getProjectTriggers();
  for (const trigger of triggers) {
    if (trigger.getHandlerFunction() === "createGoogleFormsWithMetadata") {
      ScriptApp.deleteTrigger(trigger);
    }
  }
}

// Method to generate questionnaire
function createGoogleFormsWithMetadata() {
  const BATCH_SIZE = 5
  const MAX_EXECUTION_TIME_MS = 5.5 * 60 * 1000; // 5.5 minutes
  const startTime = Date.now();
  const spreadsheet = SpreadsheetApp.openById("12RebrHoPqKhm_qoCJYtwJK4vCloGRBncDzbx6G3hPNo");
  const sheet = spreadsheet.getSheetByName("summarized_questionnaire_data"); // Make this explicit!
  
  const folderId = "1x7G4ro56dgYVf79SVXtBF2GbSIqPJSbQ";  // 🔹 Replace with your actual Google Drive Folder ID
  const masterSpreadsheet = SpreadsheetApp.getActiveSpreadsheet(); // Store responses in this sheet
  const folder = DriveApp.getFolderById(folderId);
  
  const data = sheet.getRange("A2:Q451").getValues();  // Read 450 rows from columns A to K
  const formLinks = [];
  const assignedSets = [];

  const props = PropertiesService.getScriptProperties();
  let startIndex = parseInt(props.getProperty("startIndex") || 0);
  let endIndex = Math.min(startIndex + BATCH_SIZE, data.length);

  console.log(`Start Index: ${startIndex}`);
  console.log(`End Index: ${endIndex}`);
  console.log(data);
  
  for (let i = startIndex; i < endIndex; i++) {
    if (Date.now() - startTime > MAX_EXECUTION_TIME_MS) {
      Logger.log("⚠️ Nearing execution timeout. Stopping early.");
      break;
    }
      const id = data[i][1];
      const url = data[i][2];
      const round = data[i][3];
      const roleSequence = data[i][4];
      const claim = data[i][5];
      const intent = data[i][6];
      const sources = data[i][7].split('@').map(source => `${source.trim()}`).join('.');
      var previousClaims = '';
      if (round > 0) {
          previousClaims = data[i][8].split('@').map(source => `- ${source.trim()}\n`).join('\n');
      }
      const evidence = data[i][9].split('@').map(source => `${source.trim()}`).join('.');
      const role = data[i][10];
      const roleDescription = data[i][11];
      const summarized_role_description = data[i][12]
      const summarized_sources = data[i][13]
      const summarized_evidence = data[i][14]
      const done_by = data[i][15]

      const form = FormApp.create("Claim Evaluation Form - " + (i + 1));
      const formFile = DriveApp.getFileById(form.getId());
      folder.addFile(formFile);  // Move Form to Folder

      const formUrl = form.getPublishedUrl(); // ✅ Generate Form link
      
      // 📌 Add the claim details as a Section Header
      form.setTitle("Role-Playing Claim Generation Questionnaire")
        .setDescription("Each question consists an instruction for guidance. Please answer all of the questions.");

      // Question 1: Role-Playing Consistency
      const q1 = form.addMultipleChoiceItem()
        .setRequired(true)
        .setTitle("How well does the Claim align with the Role's beliefs and intention?")
        .setHelpText(`\nRole: ${role}\n\nClaim: ${claim}\n\nRole Description:\n${summarized_role_description}\n\nIntent: ${intent}`);

      q1.setChoices([
        q1.createChoice("5 - Perfectly Consistent: The claim fully aligns with the role's beliefs, tone, and intent."),
        q1.createChoice("4 - Mostly Consistent: The claim follows the role and intent but may miss small details."),
        q1.createChoice("3 - Somewhat Consistent: The claim partly aligns but lacks key points or misrepresents intent."),
        q1.createChoice("2 - Mostly Inconsistent: The claim contradicts some role beliefs but has a weak connection."),
        q1.createChoice("1 - Completely Inconsistent: The claim opposes or has no connection to the role."),
      ])

      // Question 2: Content Relevancy
      if (round > 0) {
        const q2 = form.addMultipleChoiceItem()
          .setRequired(true)
          .setTitle("How relevant is the Claim compared to the provided sources and previous claims?")
          .setHelpText(`\nClaim: ${claim}\n\nSources:\n${summarized_sources}\n\nPrevious Claims:\n${previousClaims}`);

        q2.setChoices([
          q2.createChoice("5 - Perfectly Relevant: The claim fully integrates key facts from sources and previous claims."),
          q2.createChoice("4 - Mostly Relevant: The claim follows sources or previous claims but misses small details."),
          q2.createChoice("3 - Somewhat Relevant: The claim mentions sources or previous claims but misinterprets or lacks connections."),
          q2.createChoice("2 - Weakly Relevant: The claim has a weak or indirect connection to the sources or previous claims."),
          q2.createChoice("1 - Completely Irrelevant: The claim does not relate to any sources or previous claims."),
        ])
      } else {
        const q2 = form.addMultipleChoiceItem()
          .setRequired(true)
          .setTitle("How relevant is the Claim compared to the provided sources?")
          .setHelpText(`\nClaim: ${claim}\n\nSources:\n${summarized_sources}`);

          q2.setChoices([
            q2.createChoice("5 - Perfectly Relevant: The claim fully integrates key facts from sources."),
            q2.createChoice("4 - Mostly Relevant: The claim follows sources but misses small details."),
            q2.createChoice("3 - Somewhat Relevant: The claim mentions sources but misinterprets or lacks connections."),
            q2.createChoice("2 - Weakly Relevant: The claim has a weak or indirect connection to the sources."),
            q2.createChoice("1 - Completely Irrelevant: The claim does not relate to any sources."),
          ])
      }

      // Q3 Fluency
      const q3 = form.addMultipleChoiceItem()
        .setRequired(true)
        .setTitle("How fluent the Claim is in terms of grammar, clarity, and readability?")
        .setHelpText(`\nClaim: ${claim}`);

      q3.setChoices([
        q3.createChoice("5 - Excellent: Clear, well-written, and grammatically perfect."),
        q3.createChoice("4 - Good: Mostly correct, with minor errors that do not affect readability."),
        q3.createChoice("3 - Adequate: Readable but has noticeable errors or awkward phrasing."),
        q3.createChoice("2 - Poor: Contains multiple errors that make it harder to understand."),
        q3.createChoice("1 - Very Poor: Frequent errors make the claim difficult to comprehend."),
      ])

      // Q4 Factuality
      if (round > 0) {
          const q4 = form.addMultipleChoiceItem()
              .setRequired(true)
              .setTitle("How factually correct is the Claim?")
              .setHelpText(`\nClaim: ${claim}\n\nSources:\n${summarized_sources}\n\nPrevious Claims:\n${previousClaims}\n\nEvidence:\n${summarized_evidence}`);

          q4.setChoices([
            q4.createChoice("5 - Completely Accurate: Fully factual, with no misleading parts or missing context."),
            q4.createChoice("4 - Mostly Accurate: Mostly factual, with small mistakes or missing details that don’t change the meaning."),
            q4.createChoice("3 - Partially Accurate: Some parts are true, but others are misleading or missing key facts."),
            q4.createChoice("2 - Mostly Inaccurate: Many errors or missing key details, making it misleading."),
            q4.createChoice("1 - Completely Inaccurate: Completely false or highly misleading."),
          ])
      } else {
          const q4 = form.addMultipleChoiceItem()
              .setRequired(true)
              .setTitle("How factually correct is the Claim?")
              .setHelpText(`\nClaim: ${claim}\n\nSources:\n${summarized_sources}\n\nEvidence:\n${summarized_evidence}`);

          q4.setChoices([
            q4.createChoice("5 - Completely Accurate: Fully factual, with no misleading parts or missing context."),
            q4.createChoice("4 - Mostly Accurate: Mostly factual, with small mistakes or missing details that don’t change the meaning."),
            q4.createChoice("3 - Partially Accurate: Some parts are true, but others are misleading or missing key facts."),
            q4.createChoice("2 - Mostly Inaccurate: Many errors or missing key details, making it misleading."),
            q4.createChoice("1 - Completely Inaccurate: Completely false or highly misleading."),
          ])
      }

      // Q5: Label Assignment
      const q5 = form.addMultipleChoiceItem()
        .setRequired(true)
        .setTitle("Which label do you think is suitable for the Claim based on the evidence?")
        .setHelpText(`\nClaim: ${claim}\n\nEvidence:\n${summarized_evidence}`);

      q5.setChoices([
        q5.createChoice('True: Fully accurate.'),
        q5.createChoice('Half-True: Partially accurate, lacks important details or is misleading'),
        q5.createChoice('False: Inaccurate')
      ])

      // 📌 Automatically link responses to the same Google Sheet
      form.setDestination(FormApp.DestinationType.SPREADSHEET, masterSpreadsheet.getId());

      // ✅ Rename the response sheet
      const currentFormUrl = form.getEditUrl().replace("edit", "viewform");
      const formSheet = masterSpreadsheet.getSheets().find(s => s.getFormUrl() == currentFormUrl);
      if (formSheet) {
        formSheet.setName(id);
      }

      // ✅ Store the Google Form link in an array for batch update
      formLinks.push([formUrl]);
      assignedSets.push([`set_${Math.floor(i / 15)}`])
  }

  // ✅ Autofill Column M with the generated Google Form links
  sheet.getRange(1, 16, 1, 1).setValues([["assigned_as"]]);
  sheet.getRange(startIndex + 2, 16, formLinks.length, 1).setValues(assignedSets);
  sheet.getRange(1, 17, 1, 1).setValues([["google_form_links"]]);
  sheet.getRange(startIndex + 2, 17, formLinks.length, 1).setValues(formLinks);

  if (endIndex >= data.length) {
    props.deleteProperty("startIndex");
    deleteTriggers();
    Logger.log("✅ All forms created!");
  } else {
    props.setProperty("startIndex", endIndex.toString());
    Logger.log(`✅ Processed ${startIndex}–${endIndex - 1}.`);
  }
  
  Logger.log("✅ 450 Google Forms created with metadata and linked to one Google Sheet!");
}
