export function validateInputs(fields, values) {
  const errors = {};
  for (const field of fields) {
    const raw = values[field.id];
    const required = field.required !== false && !(field.required_unless && values.status === field.required_unless);
    if (raw == null || String(raw).trim() === "") {
      if (required) errors[field.id] = "This field is required.";
      continue;
    }
    if (field.type === "select") {
      if (!field.options.some((option) => String(option.value) === String(raw))) errors[field.id] = `Choose a valid ${field.label.toLowerCase()}.`;
    } else if (field.type === "text") {
      const value = String(raw).trim();
      if (field.min_length && value.length < field.min_length) errors[field.id] = `Enter at least ${field.min_length} characters.`;
      else if (field.max_length && value.length > field.max_length) errors[field.id] = `Enter no more than ${field.max_length} characters.`;
      else if (field.pattern && !new RegExp(field.pattern).test(value)) errors[field.id] = "Enter a valid value.";
    } else {
      const number = Number(raw);
      if (!Number.isFinite(number)) errors[field.id] = "Enter a valid number.";
      else if (number < field.min || number > field.max) errors[field.id] = `Enter a value from ${field.min} to ${field.max}.`;
      else if (field.type === "integer" && !Number.isInteger(number)) errors[field.id] = "Enter a whole number.";
      else if (field.step && Math.abs(number / field.step - Math.round(number / field.step)) > 1e-8) errors[field.id] = `Use increments of ${field.step}.`;
    }
  }
  return errors;
}
