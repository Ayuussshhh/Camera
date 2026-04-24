import { useEffect, useMemo, useState } from "react";
import {
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  FormControl,
  InputAdornment,
  MenuItem,
  Pagination,
  Select,
  Skeleton,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import { alpha, useTheme } from "@mui/material/styles";
import useMediaQuery from "@mui/material/useMediaQuery";
import SearchRoundedIcon from "@mui/icons-material/SearchRounded";
import AddRoundedIcon from "@mui/icons-material/AddRounded";
import FileDownloadRoundedIcon from "@mui/icons-material/FileDownloadRounded";
import {
  DataGrid,
  GridToolbarColumnsButton,
  GridToolbarContainer,
  GridToolbarDensitySelector,
  GridToolbarExport,
  GridToolbarFilterButton
} from "@mui/x-data-grid";

function GridPanelToolbar({
  compact = false,
  quickFilterValue,
  onQuickFilterChange,
  quickFilterPlaceholder,
  primaryAction,
  primaryActionLabel,
  secondaryAction,
  secondaryActionLabel
}) {
  const content = (
    <>
      <Stack direction="row" spacing={1} sx={{ alignItems: "center", flexWrap: "wrap" }}>
        {!compact ? <GridToolbarColumnsButton /> : null}
        {!compact ? <GridToolbarFilterButton /> : null}
        {!compact ? <GridToolbarDensitySelector /> : null}
        {!compact ? <GridToolbarExport /> : null}
      </Stack>

      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={1.25}
        sx={{ alignItems: { xs: "stretch", sm: "center" }, width: { xs: "100%", md: "auto" } }}
      >
        {onQuickFilterChange ? (
          <TextField
            placeholder={quickFilterPlaceholder}
            size="small"
            value={quickFilterValue}
            onChange={(event) => onQuickFilterChange(event.target.value)}
            sx={{
              minWidth: { xs: "100%", sm: 240 },
              "& .MuiInputBase-root": {
                backgroundColor: (theme) =>
                  alpha(theme.palette.background.default, compact ? 0.92 : 0.66)
              }
            }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchRoundedIcon fontSize="small" />
                </InputAdornment>
              )
            }}
          />
        ) : null}

        {secondaryAction ? (
          <Button
            color="secondary"
            onClick={secondaryAction}
            startIcon={<FileDownloadRoundedIcon fontSize="small" />}
            variant="outlined"
          >
            {secondaryActionLabel}
          </Button>
        ) : null}

        {primaryAction ? (
          <Button
            onClick={primaryAction}
            startIcon={<AddRoundedIcon fontSize="small" />}
            variant="contained"
          >
            {primaryActionLabel}
          </Button>
        ) : null}
      </Stack>
    </>
  );

  if (compact) {
    return (
      <Box
        sx={{
          px: { xs: 2, sm: 3 },
          py: 2,
          borderBottom: (theme) => `1px solid ${theme.palette.divider}`,
          display: "flex",
          flexDirection: { xs: "column", md: "row" },
          alignItems: { xs: "stretch", md: "center" },
          justifyContent: "space-between",
          gap: 1.5
        }}
      >
        {content}
      </Box>
    );
  }

  return (
    <GridToolbarContainer
      sx={{
        px: { xs: 2, sm: 3 },
        py: 2,
        borderBottom: (theme) => `1px solid ${theme.palette.divider}`,
        display: "flex",
        flexDirection: { xs: "column", md: "row" },
        alignItems: { xs: "stretch", md: "center" },
        justifyContent: "space-between",
        gap: 1.5
      }}
    >
      {content}
    </GridToolbarContainer>
  );
}

function MobileCard({ row, columns }) {
  const theme = useTheme();
  const titleColumn = columns.find((column) => column.field !== "actions");
  const detailColumns = columns.filter(
    (column) => column.field !== "actions" && column.field !== titleColumn?.field
  );

  const getValue = (column) => {
    if (column.renderCell) {
      return column.renderCell({ row, value: row[column.field] });
    }

    return row[column.field] ?? "-";
  };

  return (
    <Card
      sx={{
        mb: 2,
        border: `1px solid ${theme.palette.divider}`,
        transition: "all 0.2s ease-in-out",
        "&:hover": {
          borderColor: theme.palette.primary.main,
          boxShadow: `0 6px 18px ${alpha(theme.palette.primary.main, 0.12)}`
        }
      }}
    >
      <CardContent sx={{ p: 3 }}>
        {titleColumn ? (
          <Typography sx={{ fontWeight: 700, mb: 2 }} variant="subtitle1">
            {getValue(titleColumn)}
          </Typography>
        ) : null}

        <Box sx={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 2 }}>
          {detailColumns.map((column) => (
            <Box key={column.field}>
              <Typography
                color="text.secondary"
                sx={{ letterSpacing: 0.8, mb: 0.4, textTransform: "uppercase" }}
                variant="caption"
              >
                {column.headerName || column.field}
              </Typography>
              <Box sx={{ color: "text.primary", fontWeight: 500 }}>{getValue(column)}</Box>
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
}

function MobileCardList({
  rows,
  columns,
  loading,
  rowCount,
  paginationModel,
  onPaginationModelChange,
  pageSizeOptions,
  toolbar,
  emptyMessage
}) {
  const totalPages = Math.max(Math.ceil(rowCount / paginationModel.pageSize), 1);
  const currentPage = paginationModel.page + 1;

  if (loading) {
    return (
      <Box>
        {toolbar}
        <Box sx={{ p: 3 }}>
          {[1, 2, 3].map((index) => (
            <Card key={index} sx={{ mb: 2 }}>
              <CardContent sx={{ p: 3 }}>
                <Skeleton height={28} variant="text" width="55%" />
                <Box sx={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 2, mt: 1.5 }}>
                  <Skeleton height={20} variant="text" width="100%" />
                  <Skeleton height={20} variant="text" width="100%" />
                  <Skeleton height={20} variant="text" width="100%" />
                  <Skeleton height={20} variant="text" width="100%" />
                </Box>
              </CardContent>
            </Card>
          ))}
        </Box>
      </Box>
    );
  }

  if (!rows.length) {
    return (
      <Box>
        {toolbar}
        <Box sx={{ p: 4, textAlign: "center" }}>
          <Typography color="text.secondary" variant="body2">
            {emptyMessage}
          </Typography>
        </Box>
      </Box>
    );
  }

  return (
    <Box>
      {toolbar}

      <Box sx={{ p: 3, pt: 2.5 }}>
        {rows.map((row) => (
          <MobileCard columns={columns} key={row.__rowId} row={row} />
        ))}
      </Box>

      <Divider />

      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={2}
        sx={{
          p: 2.25,
          alignItems: "center",
          justifyContent: "space-between"
        }}
      >
        <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
          <Typography color="text.secondary" variant="body2">
            Rows per page
          </Typography>
          <FormControl size="small">
            <Select
              value={paginationModel.pageSize}
              onChange={(event) =>
                onPaginationModelChange({
                  page: 0,
                  pageSize: Number(event.target.value)
                })
              }
              sx={{ minWidth: 84 }}
            >
              {pageSizeOptions.map((size) => (
                <MenuItem key={size} value={size}>
                  {size}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Stack>

        <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} sx={{ alignItems: "center" }}>
          <Typography color="text.secondary" variant="body2">
            {paginationModel.page * paginationModel.pageSize + 1}-
            {Math.min((paginationModel.page + 1) * paginationModel.pageSize, rowCount)} of {rowCount}
          </Typography>
          <Pagination
            color="primary"
            count={totalPages}
            onChange={(_, page) =>
              onPaginationModelChange({
                ...paginationModel,
                page: page - 1
              })
            }
            page={currentPage}
            shape="rounded"
            showFirstButton
            showLastButton
            size="small"
          />
        </Stack>
      </Stack>
    </Box>
  );
}

export default function CustomDataGrid({
  title,
  subtitle,
  columns = [],
  rows = [],
  loading = false,
  emptyMessage = "No records found.",
  rowKey = "id",
  pageSizeOptions = [10, 25, 50],
  showToolbar = false,
  quickFilterPlaceholder = "Search records...",
  primaryAction,
  primaryActionLabel = "Add",
  secondaryAction,
  secondaryActionLabel = "Export",
  initialState,
  sx
}) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));
  const [quickFilterValue, setQuickFilterValue] = useState("");
  const [paginationModel, setPaginationModel] = useState(
    initialState?.pagination?.paginationModel || {
      page: 0,
      pageSize: pageSizeOptions[0] || 10
    }
  );

  const normalizedColumns = useMemo(
    () =>
      columns.map((column) => ({
        flex: column.flex ?? 1,
        minWidth: column.minWidth ?? 160,
        align: column.align,
        headerAlign: column.headerAlign || column.align,
        sortable: column.sortable ?? true,
        ...column,
        renderCell:
          column.renderCell ||
          (column.render
            ? (params) => column.render(params.row)
            : undefined)
      })),
    [columns]
  );

  const rowsWithIds = useMemo(
    () =>
      rows.map((row, index) => ({
        ...row,
        __rowId: row[rowKey] ?? row.id ?? `${rowKey}-${index}`
      })),
    [rowKey, rows]
  );

  const filteredRows = useMemo(() => {
    if (!quickFilterValue.trim()) {
      return rowsWithIds;
    }

    const query = quickFilterValue.toLowerCase();

    return rowsWithIds.filter((row) =>
      normalizedColumns.some((column) => {
        const rawValue =
          typeof column.searchValueGetter === "function"
            ? column.searchValueGetter(row)
            : row[column.field];

        return String(rawValue ?? "").toLowerCase().includes(query);
      })
    );
  }, [normalizedColumns, quickFilterValue, rowsWithIds]);

  const visibleRows = useMemo(() => {
    if (!isMobile) {
      return filteredRows;
    }

    const start = paginationModel.page * paginationModel.pageSize;
    return filteredRows.slice(start, start + paginationModel.pageSize);
  }, [filteredRows, isMobile, paginationModel.page, paginationModel.pageSize]);

  useEffect(() => {
    setPaginationModel((currentValue) =>
      currentValue.page === 0
        ? currentValue
        : {
            ...currentValue,
            page: 0
          }
    );
  }, [quickFilterValue]);

  return (
    <Card sx={{ height: "100%" }}>
      {title || subtitle ? (
        <CardContent sx={{ pb: 2.25 }}>
          {title ? (
            <Typography sx={{ fontSize: "1.125rem", fontWeight: 600 }} variant="h6">
              {title}
            </Typography>
          ) : null}
          {subtitle ? (
            <Typography color="text.secondary" sx={{ mt: 0.7 }} variant="body2">
              {subtitle}
            </Typography>
          ) : null}
        </CardContent>
      ) : null}

      {isMobile ? (
        <MobileCardList
          columns={normalizedColumns}
          emptyMessage={emptyMessage}
          loading={loading}
          onPaginationModelChange={setPaginationModel}
          pageSizeOptions={pageSizeOptions}
          paginationModel={paginationModel}
          rowCount={filteredRows.length}
          rows={visibleRows}
          toolbar={
            showToolbar ? (
              <GridPanelToolbar
                compact
                onQuickFilterChange={setQuickFilterValue}
                primaryAction={primaryAction}
                primaryActionLabel={primaryActionLabel}
                quickFilterPlaceholder={quickFilterPlaceholder}
                quickFilterValue={quickFilterValue}
                secondaryAction={secondaryAction}
                secondaryActionLabel={secondaryActionLabel}
              />
            ) : null
          }
        />
      ) : (
        <DataGrid
          autoHeight
          columns={normalizedColumns}
          disableRowSelectionOnClick
          getRowId={(row) => row.__rowId}
          initialState={initialState}
          loading={loading}
          pagination
          pageSizeOptions={pageSizeOptions}
          paginationModel={paginationModel}
          onPaginationModelChange={setPaginationModel}
          rows={filteredRows}
          slots={showToolbar ? { toolbar: GridPanelToolbar } : undefined}
          slotProps={
            showToolbar
              ? {
                  toolbar: {
                    quickFilterValue,
                    onQuickFilterChange: setQuickFilterValue,
                    quickFilterPlaceholder,
                    primaryAction,
                    primaryActionLabel,
                    secondaryAction,
                    secondaryActionLabel
                  }
                }
              : undefined
          }
          sx={{
            border: 0,
            minHeight: 420,
            "& .MuiDataGrid-columnHeaders": {
              backgroundColor: alpha(theme.palette.background.default, 0.9),
              borderTop: `1px solid ${theme.palette.divider}`
            },
            "& .MuiDataGrid-columnHeaderTitle": {
              fontSize: "0.82rem",
              fontWeight: 600,
              letterSpacing: 0.7,
              textTransform: "uppercase"
            },
            "& .MuiDataGrid-cell": {
              borderBottom: `1px solid ${theme.palette.divider}`,
              alignItems: "center"
            },
            "& .MuiDataGrid-toolbarContainer": {
              "& .MuiButton-root": {
                color: theme.palette.primary.main,
                fontWeight: 500
              }
            },
            "& .MuiDataGrid-row:hover": {
              backgroundColor: alpha(theme.palette.primary.main, 0.05)
            },
            "& .MuiDataGrid-footerContainer": {
              borderTop: `1px solid ${theme.palette.divider}`
            },
            "& .MuiDataGrid-overlayWrapperInner": {
              minHeight: 180
            },
            ...sx
          }}
        />
      )}
    </Card>
  );
}
