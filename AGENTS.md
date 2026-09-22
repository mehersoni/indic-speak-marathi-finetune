# Project rules — indic-speak-marathi

## File headers
Every file starts with a short comment (2–4 lines) stating what the file
does and why it's in this project. No author name, no date, no changelog.

## Comments
Comment only where the logic isn't obvious from the code itself. No
comment above every line. No comments that just restate the line below
them. Write comments the way a person would explain something to a
teammate — short, plain language. Avoid words like "robust", "leverage",
"seamlessly", "ensures" — just say what it does.

## No clutter
Don't add files, functions, classes, or config options that weren't
asked for. No try/except for errors that can't happen in this pipeline.
No logging frameworks, no unused argument-parser options, no
abstractions for hypothetical future needs. If a script does one thing,
it's one function, not a class.

## No hallucination
Before writing code that depends on a specific library's or model's
real behavior — tokenizer format, model class, function signatures,
vocabulary/token layout — check the actual installed package source,
the model repo's own files, or official docs. Don't infer this from
a similar model or a tutorial for a different project. If you can't
verify something, say so plainly in your response and leave a clearly
marked `# TODO: unverified —` comment in the code instead of guessing
and presenting it as settled.

## Code style
Write code that reads like one person wrote it in one sitting, not
generated. Consistent naming. No dead code. No placeholder functions.
No tutorial-style boilerplate. Prefer plain functions over classes
unless state genuinely needs to persist across calls. Keep functions
short enough to read on one screen.
