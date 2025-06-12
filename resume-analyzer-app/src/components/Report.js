import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Divider,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Fade,
  useTheme
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ReportProblemIcon from '@mui/icons-material/ReportProblem';

export default function Report({ result }) {
  const theme = useTheme();

  return (
    <Fade in={true} timeout={1000}>
      <Paper 
        elevation={3} 
        sx={{ 
          mt: 5, 
          p: 4,
          background: 'linear-gradient(to bottom right, #ffffff, #f8f9fa)',
          borderRadius: 2,
          transition: 'transform 0.3s ease-in-out',
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: '0 8px 16px rgba(0,0,0,0.1)'
          }
        }}
      >
        <Typography 
          variant="h5" 
          gutterBottom
          sx={{
            fontWeight: 'bold',
            color: theme.palette.primary.main,
            display: 'flex',
            alignItems: 'center',
            gap: 1
          }}
        >
          📊 Resume Match Report
        </Typography>

        <Typography variant="h6" sx={{ mt: 2, color: theme.palette.text.secondary }}>🔍 Match Score:</Typography>
        <Typography 
          variant="body1" 
          sx={{ 
            fontWeight: 'bold', 
            fontSize: '1.2rem', 
            mb: 2,
            color: theme.palette.primary.main
          }}
        >
          {result.score}% match with the Job Description
        </Typography>

        <Divider sx={{ my: 2 }} />

        <Typography variant="h6" sx={{ mt: 2, color: theme.palette.text.secondary }}>🧠 Extracted Skills:</Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
          {result.skills.map((skill, index) => (
            <Chip 
              key={index} 
              label={skill} 
              color="primary" 
              variant="outlined"
              sx={{
                transition: 'all 0.3s ease',
                '&:hover': {
                  backgroundColor: theme.palette.primary.main,
                  color: 'white',
                  transform: 'scale(1.05)'
                }
              }}
            />
          ))}
        </Box>

        <Typography variant="h6" sx={{ mt: 4, color: theme.palette.text.secondary }}>📘 Experience Summary:</Typography>
        <Typography 
          variant="body1" 
          sx={{ 
            mt: 1,
            lineHeight: 1.6,
            color: theme.palette.text.primary
          }}
        >
          {result.experience_summary}
        </Typography>

        <Typography variant="h6" sx={{ mt: 4, color: theme.palette.text.secondary }}>🎓 Education:</Typography>
        <Typography 
          variant="body1" 
          sx={{ 
            mt: 1,
            lineHeight: 1.6,
            color: theme.palette.text.primary
          }}
        >
          {result.education}
        </Typography>

        <Typography variant="h6" sx={{ mt: 4, color: theme.palette.text.secondary }}>📌 Suggested Improvements:</Typography>
        {result.improvements.length > 0 ? (
          <List>
            {result.improvements.map((item, index) => (
              <ListItem 
                key={index} 
                alignItems="flex-start"
                sx={{
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.warning.light, 0.1),
                    borderRadius: 1
                  }
                }}
              >
                <ListItemIcon>
                  <ReportProblemIcon color="warning" />
                </ListItemIcon>
                <ListItemText 
                  primary={item}
                  sx={{
                    '& .MuiListItemText-primary': {
                      color: theme.palette.text.primary,
                      lineHeight: 1.6
                    }
                  }}
                />
              </ListItem>
            ))}
          </List>
        ) : (
          <Typography 
            variant="body1" 
            sx={{ 
              mt: 1, 
              color: 'success.main',
              display: 'flex',
              alignItems: 'center',
              gap: 1
            }}
          >
            <CheckCircleIcon sx={{ verticalAlign: 'middle' }} /> 
            Your resume aligns well with the job description.
          </Typography>
        )}
      </Paper>
    </Fade>
  );
} 