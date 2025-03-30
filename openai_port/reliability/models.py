"""
Data models for reliability processing.

This module defines the data structures used for validating agent responses
and tracking reliability metrics.
"""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class SourceReliability(str, Enum):
    """
    Enum defining levels of source reliability.
    
    Attributes:
        HIGH: High reliability source
        MEDIUM: Medium reliability source
        LOW: Low reliability source
        UNKNOWN: Unknown reliability source
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ValidationPoint(BaseModel):
    """
    Model for storing validation results for a specific aspect (URLs, numbers, etc.)
    
    Attributes:
        is_suspicious: Whether the content contains suspicious elements
        feedback: Feedback on the validation
        suspicious_points: List of specific suspicious elements
        source_reliability: Reliability rating of the source
        verification_method: Method used for verification
        confidence_score: Confidence score of the validation (0.0-1.0)
    """
    is_suspicious: bool
    feedback: str
    suspicious_points: List[str] = Field(
        default_factory=list,
        description="Suspicious information raw name"
    )
    source_reliability: SourceReliability = SourceReliability.UNKNOWN
    verification_method: str
    confidence_score: float = 0.0


class ValidationResult(BaseModel):
    """
    Model for storing overall validation results across multiple aspects.
    
    Attributes:
        url_validation: Validation results for URLs
        number_validation: Validation results for numeric data
        information_validation: Validation results for general information
        code_validation: Validation results for code snippets
        any_suspicion: Whether any validation found suspicious content
        suspicious_points: List of all suspicious points found
        overall_feedback: Combined feedback from all validations
        overall_confidence: Overall confidence score (0.0-1.0)
    """
    url_validation: ValidationPoint
    number_validation: ValidationPoint
    information_validation: ValidationPoint
    code_validation: ValidationPoint
    any_suspicion: bool = False
    suspicious_points: List[str] = Field(default_factory=list)
    overall_feedback: str = ""
    overall_confidence: float = 0.0

    def calculate_suspicion(self) -> str:
        """
        Calculate overall suspicion based on individual validation results.
        
        Updates the any_suspicion, suspicious_points, overall_confidence,
        and overall_feedback fields.
        
        Returns:
            str: A summary of the validation results
        """
        self.any_suspicion = any([
            self.url_validation.is_suspicious,
            self.number_validation.is_suspicious,
            self.information_validation.is_suspicious,
            self.code_validation.is_suspicious
        ])

        self.suspicious_points = []
        validation_details = []

        # Collect URL validation details
        if self.url_validation.is_suspicious:
            self.suspicious_points.extend(self.url_validation.suspicious_points)
            validation_details.append(f"URL Issues: {self.url_validation.feedback}")
            validation_details.extend([f"- {point}" for point in self.url_validation.suspicious_points])

        # Collect number validation details
        if self.number_validation.is_suspicious:
            self.suspicious_points.extend(self.number_validation.suspicious_points)
            validation_details.append(f"Number Issues: {self.number_validation.feedback}")
            validation_details.extend([f"- {point}" for point in self.number_validation.suspicious_points])
            
        # Collect information validation details
        if self.information_validation.is_suspicious:
            self.suspicious_points.extend(self.information_validation.suspicious_points)
            validation_details.append(f"Information Issues: {self.information_validation.feedback}")
            validation_details.extend([f"- {point}" for point in self.information_validation.suspicious_points])

        # Collect code validation details
        if self.code_validation.is_suspicious:
            self.suspicious_points.extend(self.code_validation.suspicious_points)
            validation_details.append(f"Code Issues: {self.code_validation.feedback}")
            validation_details.extend([f"- {point}" for point in self.code_validation.suspicious_points])

        # Calculate overall confidence
        self.overall_confidence = sum([
            self.url_validation.confidence_score,
            self.number_validation.confidence_score,
            self.information_validation.confidence_score,
            self.code_validation.confidence_score
        ]) / 4.0

        # Generate overall feedback
        if validation_details:
            self.overall_feedback = "\n".join(validation_details)
        else:
            self.overall_feedback = "No suspicious content detected."

        # Return complete validation summary for editor
        validation_summary = [
            "Validation Summary:",
            f"Overall Confidence: {self.overall_confidence:.2f}",
            f"Suspicious Content Detected: {'Yes' if self.any_suspicion else 'No'}",
            "\nDetailed Feedback:",
            self.overall_feedback
        ]
        
        return "\n".join(validation_summary)
