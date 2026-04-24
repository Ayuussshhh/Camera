import { Button } from "@mui/material";

export function PrimaryButton({ children, ...props }) {
  return (
    <Button disableElevation variant="contained" {...props}>
      {children}
    </Button>
  );
}

export function SecondaryButton({ children, ...props }) {
  return (
    <Button variant="outlined" {...props}>
      {children}
    </Button>
  );
}
