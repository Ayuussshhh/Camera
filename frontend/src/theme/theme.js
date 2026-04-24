import { alpha, createTheme } from "@mui/material/styles";

function createShadows(mode) {
  if (mode === "light") {
    return [
      "none",
      "0px 2px 4px 0px rgba(47, 43, 61, 0.12)",
      "0px 2px 6px 0px rgba(47, 43, 61, 0.14)",
      "0px 3px 8px 0px rgba(47, 43, 61, 0.14)",
      "0px 3px 9px 0px rgba(47, 43, 61, 0.15)",
      "0px 4px 10px 0px rgba(47, 43, 61, 0.15)",
      "0px 4px 11px 0px rgba(47, 43, 61, 0.16)",
      "0px 4px 18px 0px rgba(47, 43, 61, 0.1)",
      "0px 4px 13px 0px rgba(47, 43, 61, 0.18)",
      "0px 5px 14px 0px rgba(47, 43, 61, 0.18)",
      "0px 5px 15px 0px rgba(47, 43, 61, 0.2)",
      "0px 5px 16px 0px rgba(47, 43, 61, 0.2)",
      "0px 6px 17px 0px rgba(47, 43, 61, 0.22)",
      "0px 6px 18px 0px rgba(47, 43, 61, 0.22)",
      "0px 6px 19px 0px rgba(47, 43, 61, 0.24)",
      "0px 7px 20px 0px rgba(47, 43, 61, 0.24)",
      "0px 7px 21px 0px rgba(47, 43, 61, 0.26)",
      "0px 7px 22px 0px rgba(47, 43, 61, 0.26)",
      "0px 8px 23px 0px rgba(47, 43, 61, 0.28)",
      "0px 8px 24px 6px rgba(47, 43, 61, 0.28)",
      "0px 9px 25px 0px rgba(47, 43, 61, 0.3)",
      "0px 9px 26px 0px rgba(47, 43, 61, 0.32)",
      "0px 9px 27px 0px rgba(47, 43, 61, 0.32)",
      "0px 10px 28px 0px rgba(47, 43, 61, 0.34)",
      "0px 10px 30px 0px rgba(47, 43, 61, 0.34)"
    ];
  }

  return [
    "none",
    "0px 2px 4px 0px rgba(15, 20, 34, 0.12)",
    "0px 2px 6px 0px rgba(15, 20, 34, 0.14)",
    "0px 3px 8px 0px rgba(15, 20, 34, 0.14)",
    "0px 3px 9px 0px rgba(15, 20, 34, 0.15)",
    "0px 4px 10px 0px rgba(15, 20, 34, 0.15)",
    "0px 4px 11px 0px rgba(15, 20, 34, 0.16)",
    "0px 4px 18px 0px rgba(15, 20, 34, 0.1)",
    "0px 4px 13px 0px rgba(15, 20, 34, 0.18)",
    "0px 5px 14px 0px rgba(15, 20, 34, 0.18)",
    "0px 5px 15px 0px rgba(15, 20, 34, 0.2)",
    "0px 5px 16px 0px rgba(15, 20, 34, 0.2)",
    "0px 6px 17px 0px rgba(15, 20, 34, 0.22)",
    "0px 6px 18px 0px rgba(15, 20, 34, 0.22)",
    "0px 6px 19px 0px rgba(15, 20, 34, 0.24)",
    "0px 7px 20px 0px rgba(15, 20, 34, 0.24)",
    "0px 7px 21px 0px rgba(15, 20, 34, 0.26)",
    "0px 7px 22px 0px rgba(15, 20, 34, 0.26)",
    "0px 8px 23px 0px rgba(15, 20, 34, 0.28)",
    "0px 8px 24px 6px rgba(15, 20, 34, 0.28)",
    "0px 9px 25px 0px rgba(15, 20, 34, 0.3)",
    "0px 9px 26px 0px rgba(15, 20, 34, 0.32)",
    "0px 9px 27px 0px rgba(15, 20, 34, 0.32)",
    "0px 10px 28px 0px rgba(15, 20, 34, 0.34)",
    "0px 10px 30px 0px rgba(15, 20, 34, 0.34)"
  ];
}

function getPalette(mode) {
  if (mode === "dark") {
    return {
      mode,
      primary: { light: "#5CAFFF", main: "#1777D1", dark: "#0D4F91" },
      secondary: { light: "#C9D4E3", main: "#8A97A9", dark: "#6A7585" },
      success: { light: "#86EFAC", main: "#16A34A", dark: "#166534" },
      warning: { light: "#FDBA74", main: "#EA580C", dark: "#9A3412" },
      error: { light: "#FCA5A5", main: "#DC2626", dark: "#991B1B" },
      info: { light: "#7DD3FC", main: "#0EA5E9", dark: "#0369A1" },
      background: { default: "#0D1726", paper: "#152235" },
      text: { primary: alpha("#F1F5F9", 0.9), secondary: alpha("#F1F5F9", 0.6) },
      divider: alpha("#F1F5F9", 0.12)
    };
  }

  return {
    mode,
    primary: { light: "#59A7F5", main: "#1769AA", dark: "#0E4773" },
    secondary: { light: "#D8E2EC", main: "#75869A", dark: "#4F6378" },
    success: { light: "#86EFAC", main: "#16A34A", dark: "#166534" },
    warning: { light: "#FDBA74", main: "#EA580C", dark: "#9A3412" },
    error: { light: "#FCA5A5", main: "#DC2626", dark: "#991B1B" },
    info: { light: "#7DD3FC", main: "#0EA5E9", dark: "#0369A1" },
    background: { default: "#F4F7FB", paper: "#FFFFFF" },
    text: { primary: alpha("#566A7F", 0.96), secondary: alpha("#697A8D", 0.82) },
    divider: alpha("#566A7F", 0.12)
  };
}

export function createAppTheme(mode) {
  const palette = getPalette(mode);

  return createTheme({
    palette,
    shadows: createShadows(mode),
    shape: {
      borderRadius: 2
    },
    mixins: {
      toolbar: {
        minHeight: 64
      }
    },
    typography: {
      fontFamily:
        '"Public Sans", sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial',
      fontSize: 13.125,
      h1: { fontWeight: 500, fontSize: "2.375rem", lineHeight: 1.368421 },
      h2: { fontWeight: 500, fontSize: "2rem", lineHeight: 1.375 },
      h3: { fontWeight: 500, fontSize: "1.625rem", lineHeight: 1.38462 },
      h4: { fontWeight: 500, fontSize: "1.375rem", lineHeight: 1.364 },
      h5: { fontWeight: 500, fontSize: "1.125rem", lineHeight: 1.3334 },
      h6: { fontSize: "0.9375rem", lineHeight: 1.4 },
      subtitle1: { fontSize: "1rem", letterSpacing: "0.15px" },
      subtitle2: { lineHeight: 1.32, fontSize: "0.875rem", letterSpacing: "0.1px" },
      body1: { lineHeight: 1.3, fontSize: "0.9375rem" },
      body2: { fontSize: "0.8125rem", lineHeight: 1.3 },
      button: {
        fontWeight: 500,
        lineHeight: 1.2,
        fontSize: "0.9375rem",
        letterSpacing: "0.43px",
        textTransform: "none"
      },
      caption: {
        lineHeight: 1.3,
        fontSize: "0.6875rem"
      },
      overline: {
        fontSize: "0.75rem",
        letterSpacing: "1px"
      }
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          html: {
            height: "100%"
          },
          "body, #__next": {
            minHeight: "100%"
          },
          body: {
            backgroundColor: palette.background.default,
            backgroundImage:
              mode === "light"
                ? `radial-gradient(circle at top right, ${alpha(
                    palette.primary.main,
                    0.08
                  )} 0%, transparent 28%)`
                : `radial-gradient(circle at top right, ${alpha(
                    palette.primary.light,
                    0.12
                  )} 0%, transparent 26%)`
          }
        }
      },
      MuiButton: {
        styleOverrides: {
          root: {
            minWidth: 50,
            borderRadius: 2,
            minHeight: 42,
            paddingInline: 18,
            transition: "all 0.2s ease",
            "&:not(.Mui-disabled):active": {
              transform: "scale(0.98)"
            }
          },
          contained: {
            boxShadow: `0px 4px 12px ${alpha(palette.primary.main, 0.28)}`,
            "&:hover": {
              boxShadow: `0px 4px 12px ${alpha(palette.primary.main, 0.28)}`
            }
          },
          outlined: {
            borderColor: alpha(palette.text.primary, 0.18),
            "&:hover": {
              borderColor: palette.primary.main,
              backgroundColor: alpha(palette.primary.main, 0.05)
            }
          }
        }
      },
      MuiCard: {
        defaultProps: {
          elevation: 7
        },
        styleOverrides: {
          root: {
            borderRadius: 2,
            border: `1px solid ${palette.divider}`,
            backgroundImage: "none",
            boxShadow: "0px 2px 10px rgba(67, 89, 113, 0.08)"
          }
        }
      },
      MuiCardHeader: {
        styleOverrides: {
          root: {
            padding: "24px",
            "& + .MuiCardContent-root": {
              paddingTop: 0
            }
          },
          title: {
            fontWeight: 500,
            lineHeight: 1.334,
            letterSpacing: "0.15px",
            fontSize: "1.125rem"
          }
        }
      },
      MuiCardContent: {
        styleOverrides: {
          root: {
            padding: "24px",
            "&:last-of-type": {
              paddingBottom: "24px"
            }
          }
        }
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: "none"
          }
        }
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: "transparent",
            boxShadow: "none"
          }
        }
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: palette.background.paper,
            borderRight: `1px solid ${palette.divider}`,
            boxShadow: "0px 2px 12px rgba(67, 89, 113, 0.08)"
          }
        }
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 2,
            fontWeight: 500
          }
        }
      },
      MuiOutlinedInput: {
        styleOverrides: {
          root: {
            borderRadius: 2,
            "&:hover .MuiOutlinedInput-notchedOutline": {
              borderColor: alpha(palette.primary.main, 0.4)
            },
            "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
              borderWidth: 1,
              borderColor: palette.primary.main
            }
          }
        }
      },
      MuiBottomNavigation: {
        styleOverrides: {
          root: {
            minHeight: 72,
            backgroundColor: palette.background.paper,
            borderTop: `1px solid ${palette.divider}`
          }
        }
      },
      MuiTableHead: {
        styleOverrides: {
          root: {
            textTransform: "uppercase"
          }
        }
      },
      MuiTableCell: {
        styleOverrides: {
          root: {
            borderBottom: `1px solid ${palette.divider}`
          },
          head: {
            fontWeight: 500,
            fontSize: "0.8125rem",
            letterSpacing: "1px",
            textTransform: "uppercase",
            color: palette.text.secondary,
            whiteSpace: "nowrap"
          },
          body: {
            letterSpacing: "0.25px",
            color: palette.text.secondary
          }
        }
      },
      MuiTableRow: {
        styleOverrides: {
          root: {
            "& .MuiTableCell-head:not(.MuiTableCell-paddingCheckbox):first-of-type, & .MuiTableCell-root:not(.MuiTableCell-paddingCheckbox):first-of-type":
              {
                paddingLeft: 24
              },
            "& .MuiTableCell-head:last-child, & .MuiTableCell-root:last-child": {
              paddingRight: 24
            },
            "&:hover": {
              backgroundColor: alpha(palette.primary.main, 0.045)
            }
          }
        }
      },
      MuiDataGrid: {
        styleOverrides: {
          root: {
            border: 0
          }
        }
      },
      MuiToolbar: {
        styleOverrides: {
          root: {
            minHeight: "64px !important"
          }
        }
      },
      MuiLinearProgress: {
        styleOverrides: {
          root: {
            borderRadius: 999,
            overflow: "hidden"
          }
        }
      },
      MuiDialog: {
        styleOverrides: {
          paper: {
            borderRadius: 2
          }
        }
      }
    }
  });
}
